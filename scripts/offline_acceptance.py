#!/usr/bin/env python3
"""Final offline acceptance test (assignment section 8).

Run AFTER installing Sellbird and pulling the model, with the internet disconnected:

    python scripts/offline_acceptance.py --max-prospects 4

What it does
  * installs a socket-level guard: any non-loopback connection or DNS lookup raises and is logged
  * runs all 8 local AI workflows end to end on a FRESH database
  * verifies: every inference call went to loopback, cost is $0, no blocked external attempt happened
  * writes evidence/offline_acceptance_<timestamp>.json and .md

Exit code 0 only if every workflow succeeded on local inference.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sellbird.local_sales import netguard  # noqa: E402

netguard.block_external_network()  # must happen before anything else can open a socket

from sellbird.local_sales import workflows as wf  # noqa: E402
from sellbird.local_sales.embeddings import Embedder  # noqa: E402
from sellbird.local_sales.hardware import PROFILES, recommend_model  # noqa: E402
from sellbird.local_sales.llm import LocalLLM, read_inference_log  # noqa: E402
from sellbird.local_sales.store import Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Ollama tag (default: SELLBIRD_LOCAL_MODEL / HARVEY_LOCAL_MODEL or qwen3:1.7b)")
    ap.add_argument("--max-prospects", type=int, default=4)
    ap.add_argument("--db", default=str(ROOT / "data" / "local_sales_acceptance.db"))
    a = ap.parse_args()

    rec = recommend_model()
    prof = PROFILES[rec["profile"]]
    llm = LocalLLM(model=a.model, num_ctx=prof["num_ctx"], num_predict=prof["num_predict"])
    health = llm.health()
    print(f"[hardware] {rec['hardware']['ram_gb']} GB RAM, {rec['hardware']['cpu_cores']} cores, "
          f"GPU={bool(rec['hardware']['gpus'])} -> recommended {rec['recommended_model']} ({rec['profile']})")
    print(f"[ollama] {health}")
    if not health["ok"]:
        print("\nABORT: local Ollama is not ready. Not falling back to any cloud service.\n" + health["error"])
        return 2

    db = Path(a.db)
    if db.exists():
        db.unlink()
    ctx = wf.Ctx(Store(db), llm, Embedder())
    started = time.time()
    results: dict = {}

    def step(n, name, fn):
        t = time.time()
        try:
            out = fn()
            ok = out.get("status", "ok") == "ok" if isinstance(out, dict) else True
        except Exception as e:  # report, keep going so the evidence shows everything
            out, ok = {"error": repr(e)}, False
        results[name] = {"step": n, "ok": ok, "seconds": round(time.time() - t, 1), "output": out}
        print(f"[{n}] {name:<34} {'PASS' if ok else 'FAIL'}  ({results[name]['seconds']}s)")

    ingest = wf.ingest_company_files(ctx.store)
    wf.load_prospects(ctx.store)
    products = ctx.store.products()
    step(1, "business_profile", lambda: wf.run_workflow(ctx, "business_profile", {}))
    for i, p in enumerate(products):
        step(2, f"icp::{p['name']}", lambda p=p: wf.run_workflow(ctx, "icp", {"product_id": p["id"]}))
    prospects = ctx.store.prospects()
    # Prospect data comes from the previously downloaded local file (step 3 = research from local data)
    chosen = prospects[: a.max_prospects]
    prod = ctx.store.products()[0]
    for pr in chosen:
        pl = {"prospect_id": pr["id"], "product_id": prod["id"]}
        step(3, f"research::{pr['company']}", lambda pl=pl: wf.run_workflow(ctx, "research", pl))
        step(4, f"signals::{pr['company']}", lambda pl=pl: wf.run_workflow(ctx, "signals", pl))
        step(5, f"problem::{pr['company']}", lambda pl=pl: wf.run_workflow(ctx, "problem", pl))
        step(7, f"score::{pr['company']}", lambda pl=pl: wf.run_workflow(ctx, "score", pl))
    ranked = ctx.store.q("SELECT s.total,s.tier,p.id pid,p.company FROM scores s JOIN prospects p ON p.id=s.prospect_id "
                         "WHERE s.product_id=? ORDER BY s.total DESC", (prod["id"],))
    for r in ranked[:2]:
        step(6, f"email::{r['company']}", lambda r=r: wf.run_workflow(ctx, "email", {"prospect_id": r["pid"], "product_id": prod["id"]}))
    ev = wf.evaluate_reply_classifier(ctx)
    results["reply_classification"] = {"step": 8, "ok": ev["total"] > 0 and all(x["status"] == "ok" for x in ev["results"]),
                                       "accuracy_vs_human_labels": ev["accuracy"], "output": ev}
    print(f"[8] reply_classification accuracy vs labels: {ev['accuracy']} ({ev['correct']}/{ev['total']})")
    print("    per class:", ev["per_class_correct"])
    for m in ev["misclassified"]:
        print(f"    miss: expected {m['expected']:<16} got {m['predicted']:<16} | {m['text'][:70]}")

    # ---- proof section
    log = [r for r in read_inference_log(100000) if r["ts"] >= started]
    guard = netguard.report()
    proof = {
        "inference_calls": len(log),
        "all_endpoints_loopback": all(r["endpoint"].startswith(("http://127.0.0.1", "http://localhost")) for r in log),
        "distinct_endpoints": sorted({r["endpoint"] for r in log}),
        "distinct_models_served": sorted({str(r.get("model_served")) for r in log}),
        "total_cost_usd": sum(r["cost_usd"] for r in log),
        "external_connection_attempts_blocked": len(guard["blocked_attempts"]),
        "blocked_attempts": guard["blocked_attempts"],
        "embedding_backend": ctx.embedder.backend,
    }
    all_ok = all(v["ok"] for v in results.values()) and proof["inference_calls"] > 0 \
        and proof["all_endpoints_loopback"] and proof["total_cost_usd"] == 0 \
        and proof["external_connection_attempts_blocked"] == 0
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "hardware": rec, "model": llm.model,
              "ollama_health": health, "company": ingest["business"], "workflows": results,
              "ranking": ranked, "proof": proof, "result": "PASS" if all_ok else "FAIL"}
    out = ROOT / "evidence"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"offline_acceptance_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    md = [f"# Offline acceptance — {report['timestamp']}", f"**Result: {report['result']}**  ·  model `{llm.model}`  ·  "
          f"embeddings `{proof['embedding_backend']}`", "", "| step | workflow | result | seconds |", "|---|---|---|---|"]
    md += [f"| {v['step']} | {k} | {'PASS' if v['ok'] else 'FAIL'} | {v.get('seconds', '-')} |" for k, v in results.items()]
    md += ["", "## Proof of local-only inference", f"- inference calls: {proof['inference_calls']}",
           f"- endpoints: {proof['distinct_endpoints']}", f"- models served: {proof['distinct_models_served']}",
           f"- total cost: ${proof['total_cost_usd']:.2f}",
           f"- external connection attempts blocked by netguard: {proof['external_connection_attempts_blocked']}",
           "", "## Ranking", *[f"- {r['company']}: {r['total']} (tier {r['tier']})" for r in ranked]]
    (out / f"offline_acceptance_{stamp}.md").write_text("\n".join(md))
    print(f"\nRESULT: {report['result']}  | evidence/offline_acceptance_{stamp}.(json|md)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
