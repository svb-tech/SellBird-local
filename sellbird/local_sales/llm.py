"""Local-only LLM client (Ollama).

Guarantees:
  * Only talks to a loopback address (or a private-LAN address if the operator
    explicitly sets LOCAL_AI_ALLOW_PRIVATE_HOST=1 for their own server).
  * Never falls back to any cloud API. If Ollama is down it raises
    LocalAIUnavailable and callers queue the task or use deterministic rules.
  * Every call is appended to logs/inference.jsonl (model, tokens, latency,
    endpoint, cost=0) so local inference is traceable.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from sellbird.paths import PROJECT_ROOT

DEFAULT_HOST = "http://127.0.0.1:11434"
INFERENCE_LOG = PROJECT_ROOT / "logs" / "inference.jsonl"
_log_lock = threading.Lock()


class LocalAIUnavailable(Exception):
    """Ollama (or the requested model) cannot be used. Never triggers a cloud fallback."""


class NonLocalEndpointError(LocalAIUnavailable):
    """OLLAMA_HOST points somewhere that is not this machine / the operator's LAN."""


def normalise_host(raw: str | None = None) -> str:
    host = (raw or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).strip()
    if "://" not in host:
        host = "http://" + host
    p = urlparse(host)
    if not p.port:
        host = f"{p.scheme}://{p.hostname}:11434"
    return host.rstrip("/")


def assert_local_endpoint(host: str) -> None:
    name = urlparse(host).hostname or ""
    if name in ("localhost", "ip6-localhost"):
        return
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        raise NonLocalEndpointError(
            f"Refusing OLLAMA_HOST '{host}': hostnames other than localhost are not allowed."
        )
    if ip.is_loopback:
        return
    if ip.is_private and os.getenv("LOCAL_AI_ALLOW_PRIVATE_HOST") == "1":
        return
    raise NonLocalEndpointError(
        f"Refusing OLLAMA_HOST '{host}': inference must stay on this machine "
        "(set LOCAL_AI_ALLOW_PRIVATE_HOST=1 only for your own LAN server)."
    )


def _write_log(record: dict) -> None:
    INFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, INFERENCE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_inference_log(limit: int = 50) -> list[dict]:
    if not INFERENCE_LOG.exists():
        return []
    lines = INFERENCE_LOG.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


class LocalLLM:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        num_ctx: int = 4096,
        num_predict: int = 700,
        temperature: float = 0.2,
        timeout: float = 180.0,
        seed: int = 7,
    ):
        self.host = normalise_host(host)
        assert_local_endpoint(self.host)
        self.model = model or os.getenv("SELLBIRD_LOCAL_MODEL") or os.getenv("HARVEY_LOCAL_MODEL") or "qwen3:1.7b"
        self.options = {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "seed": seed,
        }
        self.timeout = timeout

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=4.0)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:
            return {
                "ok": False, "host": self.host, "model": self.model, "installed_models": [],
                "error": f"Cannot reach local Ollama at {self.host}: {e}. Start it with `ollama serve`.",
            }
        if self.model not in names:
            return {
                "ok": False, "host": self.host, "model": self.model, "installed_models": names,
                "error": f"Model '{self.model}' is not installed. Run: ollama pull {self.model}",
            }
        return {"ok": True, "host": self.host, "model": self.model,
                "installed_models": names, "error": ""}

    # -------------------------------------------------------------- inference
    def _chat(self, workflow: str, system: str, user: str, schema: dict | None) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": f"[workflow:{workflow}]\n{system}"},
                {"role": "user", "content": user},
            ],
            "options": self.options,
        }
        if schema is not None:
            payload["format"] = schema
        call_id = uuid.uuid4().hex[:12]
        started = time.time()
        rec = {
            "id": call_id, "ts": started, "workflow": workflow, "provider": "ollama-local",
            "endpoint": f"{self.host}/api/chat", "model_requested": self.model,
            "prompt_chars": len(system) + len(user), "cost_usd": 0.0, "success": False,
        }
        try:
            r = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            rec.update(error=str(e)[:300], latency_s=round(time.time() - started, 3))
            _write_log(rec)
            raise LocalAIUnavailable(
                f"Local Ollama call failed ({self.host}): {e}. No cloud fallback is used."
            ) from e
        text = ((data.get("message") or {}).get("content") or "").strip()
        eval_count = data.get("eval_count") or 0
        eval_ns = data.get("eval_duration") or 0
        rec.update(
            success=bool(text), model_served=data.get("model"),
            prompt_tokens=data.get("prompt_eval_count"), completion_tokens=eval_count,
            tokens_per_sec=round(eval_count / (eval_ns / 1e9), 2) if eval_ns else None,
            latency_s=round(time.time() - started, 3), response_chars=len(text),
        )
        _write_log(rec)
        if not text:
            raise LocalAIUnavailable("Ollama returned an empty response.")
        return text, rec

    @staticmethod
    def _parse_json(text: str):
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except ValueError:
            pass
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s : e + 1])
            except ValueError:
                return None
        return None

    def generate_json(self, workflow: str, system: str, user: str, schema: dict,
                      retries: int = 2) -> tuple[dict, dict]:
        """Schema-constrained JSON generation with bounded repair retries."""
        last = ""
        for attempt in range(retries + 1):
            prompt = user if attempt == 0 else (
                user + "\n\nYour previous reply was not valid JSON for the schema. "
                "Reply again with ONLY the JSON object."
            )
            text, rec = self._chat(workflow, system, prompt, schema)
            parsed = self._parse_json(text)
            if isinstance(parsed, dict):
                rec["attempts"] = attempt + 1
                return parsed, rec
            last = text[:200]
        raise ValueError(f"Model did not return valid JSON after {retries + 1} attempts: {last}")
