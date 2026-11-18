"""``brain.provider: ollama`` — Sellbird's Brain thinking on a local Ollama.

Plugs into Brain's public ``transport`` seam (see sellbird/brain.py), so the
whole Sellbird pipeline (writer, replies, research) can run on a local model
instead of Claude or Codex. Guarantees, same as sellbird.local_sales.llm:

  * the endpoint must be this machine (loopback), or your own LAN server when
    LOCAL_AI_ALLOW_PRIVATE_HOST=1; anything else is refused at start-up;
  * there is NO fallback to Claude, Codex or any cloud API. If Ollama is down
    the call returns "" (callers already handle that), ``paused_reason``
    explains why, and the next call retries;
  * every call is appended to logs/inference.jsonl with cost_usd = 0.
"""
from __future__ import annotations

import os
import re
import time
import uuid

import httpx

from sellbird.local_sales.llm import (
    LocalAIUnavailable,
    _write_log,
    assert_local_endpoint,
    normalise_host,
)

_RETRY_AFTER_SECONDS = 20.0


def default_model() -> str:
    """Explicit env override, else the model that fits this machine."""
    env = os.getenv("SELLBIRD_LOCAL_MODEL") or os.getenv("HARVEY_LOCAL_MODEL")
    if env:
        return env
    try:
        from sellbird.local_sales.hardware import recommend_model

        return recommend_model()["recommended_model"]
    except Exception:
        return "qwen3:1.7b"


class OllamaTransport:
    """``async transport(prompt, *, agent, task, timeout, expect_json) -> str``."""

    def __init__(self, model: str = "", host: str | None = None, num_ctx: int = 4096,
                 num_predict: int = 900, temperature: float = 0.3):
        self.host = normalise_host(host)
        assert_local_endpoint(self.host)  # raises NonLocalEndpointError
        self.model = model or default_model()
        self.options = {"num_ctx": num_ctx, "num_predict": num_predict,
                        "temperature": temperature}
        self._last_error = ""
        self._last_error_at = 0.0

    @property
    def paused_reason(self) -> str:
        """Why calls are being skipped. Clears after a short wait so Ollama
        coming back up is noticed without restarting Sellbird."""
        if self._last_error and time.monotonic() - self._last_error_at < _RETRY_AFTER_SECONDS:
            return self._last_error
        return ""

    async def __call__(self, prompt: str, *, agent: str = "", task: str = "",
                       timeout: float = 180.0, expect_json: bool = False) -> str:
        payload = {
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": self.options,
        }
        if expect_json:
            payload["format"] = "json"
        started = time.time()
        rec = {
            "id": uuid.uuid4().hex[:12], "ts": started,
            "workflow": f"brain:{agent or 'agent'}/{task or 'think'}",
            "provider": "ollama-local", "endpoint": f"{self.host}/api/chat",
            "model_requested": self.model, "prompt_chars": len(prompt),
            "cost_usd": 0.0, "success": False,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{self.host}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            self._last_error = (
                f"Local Ollama unavailable at {self.host} ({type(e).__name__}). Start it with "
                f"`ollama serve` and `ollama pull {self.model}`. No cloud fallback is used.")
            self._last_error_at = time.monotonic()
            rec.update(error=str(e)[:300], latency_s=round(time.time() - started, 3))
            _write_log(rec)
            raise LocalAIUnavailable(self._last_error) from e
        self._last_error = ""
        text = ((data.get("message") or {}).get("content") or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        evals = data.get("eval_count") or 0
        eval_ns = data.get("eval_duration") or 0
        rec.update(
            success=bool(text), model_served=data.get("model"),
            prompt_tokens=data.get("prompt_eval_count"), completion_tokens=evals,
            tokens_per_sec=round(evals / (eval_ns / 1e9), 2) if eval_ns else None,
            latency_s=round(time.time() - started, 3), response_chars=len(text))
        _write_log(rec)
        return text


class NoQuota:
    """Ollama has no subscription window: nothing to meter, nothing to pay."""

    async def get_utilization(self):
        return None
