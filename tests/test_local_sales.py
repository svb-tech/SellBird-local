import json
import pytest

from sellbird.local_sales import llm as llm_mod
from sellbird.local_sales import workflows as wf
from sellbird.local_sales.embeddings import Embedder
from sellbird.local_sales.hardware import Hardware, recommend_model
from sellbird.local_sales.llm import LocalAIUnavailable, LocalLLM, NonLocalEndpointError
from sellbird.local_sales.store import Store
from tests import fake_ollama


@pytest.fixture()
def server():
    srv = fake_ollama.start()
    yield srv
    srv.shutdown()


@pytest.fixture()
def ctx(tmp_path, server, monkeypatch):
    monkeypatch.setattr(llm_mod, "INFERENCE_LOG", tmp_path / "inference.jsonl")
    monkeypatch.setenv("SELLBIRD_DISABLE_ST", "1")
    llm = LocalLLM(host=f"http://127.0.0.1:{server.server_port}", timeout=10)
    return wf.Ctx(Store(tmp_path / "t.db"), llm, Embedder())


def _prepare(ctx):
    wf.ingest_company_files(ctx.store)
    wf.load_prospects(ctx.store)
    pid = ctx.store.products()[0]["id"]
    assert wf.run_workflow(ctx, "business_profile", {})["status"] == "ok"
    assert wf.run_workflow(ctx, "icp", {"product_id": pid})["status"] == "ok"
    return pid


def test_hardware_recommendation_tiers():
    laptop = Hardware("x", "cpu", 4, 8.0)
    assert recommend_model(laptop)["recommended_model"] == "qwen3:4b"  # 8 GB RAM -> 4 GB budget
    assert recommend_model(laptop)["profile"] == "standard"
    tiny = Hardware("x", "cpu", 2, 4.0)
    assert recommend_model(tiny)["recommended_model"] == "qwen3:1.7b"
    big = Hardware("x", "cpu", 16, 64.0, gpus=[{"name": "g", "vram_gb": 24.0}], vram_gb=24.0)
    r = recommend_model(big)
    assert r["profile"] == "high_performance" and r["recommended_model"] == "qwen3:14b"  # 19GB*1.3 > 24GB
    huge = Hardware("x", "cpu", 16, 64.0, gpus=[{"name": "g", "vram_gb": 32.0}], vram_gb=32.0)
    assert recommend_model(huge)["recommended_model"] == "qwen3:30b"
    cpu_big = Hardware("x", "cpu", 16, 64.0)
    assert recommend_model(cpu_big)["recommended_model"] == "qwen3:14b"  # CPU-only cap


def test_refuses_non_local_endpoint():
    with pytest.raises(NonLocalEndpointError):
        LocalLLM(host="http://api.openai.com")
    with pytest.raises(NonLocalEndpointError):
        LocalLLM(host="http://8.8.8.8:11434")


def test_full_pipeline_and_logging(ctx):
    pid = _prepare(ctx)
    out = wf.run_pipeline(ctx, pid, max_prospects=3)
    assert len(out["ranking"]) == 3
    assert all(r["tier"] in "ABC" for r in out["ranking"])
    log = llm_mod.read_inference_log()
    assert log and all(r["cost_usd"] == 0.0 and r["endpoint"].startswith("http://127.0.0.1") for r in log)
    assert {"business_profile", "icp", "research", "signals", "score_fit"} <= {r["workflow"] for r in log}


def test_ungrounded_signal_is_not_scored(ctx):
    pid = _prepare(ctx)
    pr = ctx.store.prospects()[1]  # Brightwell: ransomware text
    res = wf.run_workflow(ctx, "signals", {"prospect_id": pr["id"], "product_id": pid})
    assert res["ungrounded_signals"] == 1
    grounded = [s for s in res["signals"] if s["grounded"]]
    assert len(grounded) == 1 and grounded[0]["type"] == "security_incident"


def test_email_checks_and_signature(ctx):
    pid = _prepare(ctx)
    pr = ctx.store.prospects()[1]
    pl = {"prospect_id": pr["id"], "product_id": pid}
    wf.run_workflow(ctx, "signals", pl)
    wf.run_workflow(ctx, "problem", pl)
    e = wf.run_workflow(ctx, "email", pl)
    assert e["status"] == "ok" and e["delivery"].startswith("draft")
    assert e["checks"]["passed"], e["checks"]
    assert e["body"].startswith("Hi Daniel") and "Business Development" in e["body"]


def test_no_signal_means_no_email(ctx):
    pid = _prepare(ctx)
    bakery = next(p for p in ctx.store.prospects() if "Bakery" in p["company"])
    pl = {"prospect_id": bakery["id"], "product_id": pid}
    wf.run_workflow(ctx, "signals", pl)
    assert wf.run_workflow(ctx, "email", pl)["email"] is None


def test_reply_rules_are_deterministic_for_unsubscribe_and_ooo(ctx):
    assert wf.run_workflow(ctx, "reply", {"text": "Please unsubscribe me"})["method"] == "rules"
    assert wf.run_workflow(ctx, "reply", {"text": "I am out of office until Monday"})["category"] == "out_of_office"


def test_ollama_down_queues_and_uses_labelled_fallback(tmp_path, monkeypatch, server):
    monkeypatch.setattr(llm_mod, "INFERENCE_LOG", tmp_path / "inference.jsonl")
    monkeypatch.setenv("SELLBIRD_DISABLE_ST", "1")
    good = LocalLLM(host=f"http://127.0.0.1:{server.server_port}", timeout=5)
    c = wf.Ctx(Store(tmp_path / "t.db"), good, Embedder())
    pid = _prepare(c)
    server.shutdown()  # Ollama "goes down"
    c.llm.timeout = 2
    h = c.llm.health()
    assert not h["ok"] and "ollama serve" in h["error"]
    r = wf.run_workflow(c, "research", {"prospect_id": 1})
    assert r["status"] == "queued" and "cloud" in r["error"].lower()
    s = wf.run_workflow(c, "signals", {"prospect_id": 1, "product_id": pid})
    assert s["status"] == "degraded" and s["method"] == "rules"
    assert all(x["method"] == "rules" for x in s["signals"])
    assert len(c.store.queue("queued")) == 2
    rep = wf.run_workflow(c, "reply", {"text": "Sounds good, send more info"})
    assert rep["status"] == "degraded" and rep["method"] == "rules-fallback"
    sc = wf.run_workflow(c, "score", {"prospect_id": 1, "product_id": pid})
    assert sc["status"] == "degraded" and "rescaled" in sc["breakdown"]["_llm_component"]


def test_missing_prerequisites_are_reported_not_crashing(ctx):
    wf.ingest_company_files(ctx.store)
    wf.load_prospects(ctx.store)
    pid = ctx.store.products()[0]["id"]
    assert wf.run_pipeline(ctx, pid)["status"] == "blocked"
    assert wf.run_workflow(ctx, "score", {"prospect_id": 1, "product_id": pid})["status"] == "blocked"


def test_reply_eval_set_is_balanced_and_not_leaked_into_the_prompt():
    from pathlib import Path
    rows = json.loads((Path(wf.DATA_DIR) / "sample_replies.json").read_text(encoding="utf-8"))["replies"]
    per = {c: sum(r["expected"] == c for r in rows) for c in wf.REPLY_CATEGORIES}
    assert len(rows) >= 36 and min(per.values()) >= 4, per
    assert all(r["text"] not in wf.REPLY_SYSTEM_PROMPT for r in rows)  # few-shot examples are different texts


def test_reply_evaluation_reports_confusion_and_misses(ctx):
    ev = wf.evaluate_reply_classifier(ctx)
    assert ev["total"] == sum(sum(v.values()) for v in ev["confusion"].values())
    assert set(ev["per_class_correct"]) == set(wf.REPLY_CATEGORIES)
    assert len(ev["misclassified"]) == ev["total"] - ev["correct"]
