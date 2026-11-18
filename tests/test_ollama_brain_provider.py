"""brain.provider: ollama — local-only inference, never a cloud fallback.

Uses a TEST-ONLY loopback HTTP stub in place of Ollama; it is never part of the product.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.local_sales import llm as llm_mod
from sellbird.local_sales.brain_transport import OllamaTransport
from sellbird.local_sales.llm import LocalAIUnavailable, NonLocalEndpointError
from sellbird.state import StateManager


class _Stub(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append(body)
        reply = {"message": {"content": '<think>x</think>{"ok": true}'}, "model": body["model"],
                 "prompt_eval_count": 7, "eval_count": 5, "eval_duration": 10**9}
        raw = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture()
def stub():
    _Stub.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


@pytest.fixture(autouse=True)
def _log(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "INFERENCE_LOG", tmp_path / "inference.jsonl")
    return tmp_path / "inference.jsonl"


def test_config_accepts_ollama_and_model_tag():
    cfg = BrainConfig(provider="ollama", model="qwen3:4b")
    assert (cfg.provider, cfg.model) == ("ollama", "qwen3:4b")
    with pytest.raises(ValueError):
        BrainConfig(provider="openai")


def test_non_local_endpoint_is_refused():
    with pytest.raises(NonLocalEndpointError):
        OllamaTransport(model="qwen3:4b", host="http://api.openai.com:443")


@pytest.mark.asyncio
async def test_brain_thinks_on_local_ollama_and_logs_zero_cost(tmp_path, stub, _log, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{stub.server_port}")
    brain = Brain(StateManager(str(tmp_path / "s.db")), BrainConfig(provider="ollama", model="qwen3:4b"))
    assert brain.provider_label.startswith("Ollama")
    out = await brain.think("say json", expect_json=True, agent="writer", task="draft")
    assert out == '{"ok": true}'  # <think> block stripped
    assert _Stub.seen[0]["format"] == "json" and _Stub.seen[0]["model"] == "qwen3:4b"
    rec = json.loads(_log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["provider"] == "ollama-local" and rec["cost_usd"] == 0.0
    assert rec["endpoint"].startswith("http://127.0.0.1:")
    assert await brain.is_within_budget(max_percent=50) is True  # nothing to meter


@pytest.mark.asyncio
async def test_ollama_down_returns_empty_and_reports_no_cloud_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    brain = Brain(StateManager(str(tmp_path / "s.db")), BrainConfig(provider="ollama", model="qwen3:4b"))
    assert await brain.think("hello", timeout=5) == ""
    assert "No cloud fallback" in brain.paused_reason
    with pytest.raises(LocalAIUnavailable):
        await brain.transport("hello", agent="a", task="t", timeout=5, expect_json=False)


def test_local_routes_sit_behind_the_dashboard_token(tmp_path):
    from fastapi.testclient import TestClient
    from sellbird import dashboard

    (tmp_path / "data").mkdir()
    dashboard.token_store.configure(tmp_path / "data" / "dashboard-token")
    token = dashboard.token_store.get()
    origin = "http://127.0.0.1:5555"
    with TestClient(dashboard.app, base_url=origin) as anon:
        assert anon.get("/api/local/status").status_code in (401, 403)  # no token, no data
        shell = anon.get("/static/local.html")
        assert shell.status_code == 200 and "Local Sales Agent" in shell.text
    with TestClient(dashboard.app, base_url=origin,
                    headers={"Authorization": f"Bearer {token}"}) as authed:
        ok = authed.get("/api/local/status")
        assert ok.status_code == 200 and "ollama" in ok.json()
