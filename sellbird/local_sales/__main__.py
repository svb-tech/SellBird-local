"""CLI: python -m sellbird.local_sales <command>

  hardware       detect RAM/CPU/GPU and recommend a model
  serve          start the local dashboard (http://127.0.0.1:8765)
  audit          static + environment audit proving no paid/cloud AI dependency
  queue          process queued tasks (after Ollama is back)
"""
import argparse
import json
import re
import sys
from pathlib import Path

from sellbird.paths import PROJECT_ROOT

FORBIDDEN_HOSTS = ["api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
                   "api.cohere", "api.mistral.ai", "api.groq.com", "openrouter.ai", "bedrock",
                   "api.together.xyz", "api.perplexity.ai"]
FORBIDDEN_ENV = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]
FORBIDDEN_IMPORTS = r"^\s*(import|from)\s+(openai|anthropic|google\.generativeai|cohere|litellm|langchain_openai)\b"


def audit() -> dict:
    import os

    pkg = PROJECT_ROOT / "sellbird" / "local_sales"
    findings = []
    for f in sorted(pkg.glob("*.py")):
        if f.name == "__main__.py":
            continue  # this file lists the forbidden names by design
        text = f.read_text(encoding="utf-8")
        for host in FORBIDDEN_HOSTS:
            if host in text:
                findings.append(f"{f.name}: references {host}")
        for i, line in enumerate(text.splitlines(), 1):
            if re.match(FORBIDDEN_IMPORTS, line):
                findings.append(f"{f.name}:{i}: imports a hosted-LLM SDK")
    env_hits = [k for k in FORBIDDEN_ENV if os.getenv(k)]
    endpoints = {r.get("endpoint", "")[:30] for r in __import__("sellbird.local_sales.llm", fromlist=["x"]).read_inference_log(100000)}
    return {"scope": "sellbird/local_sales/*.py", "forbidden_hosts_checked": FORBIDDEN_HOSTS,
            "source_findings": findings, "paid_ai_env_vars_set": env_hits,
            "distinct_inference_endpoints_in_log": sorted(endpoints),
            "result": "PASS" if not findings and not env_hits else "FAIL"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sellbird.local_sales")
    ap.add_argument("command", choices=["hardware", "serve", "audit", "queue", "acceptance"])
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if a.command == "hardware":
        from sellbird.local_sales.hardware import recommend_model
        print(json.dumps(recommend_model(), indent=2))
    elif a.command == "audit":
        r = audit()
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["result"] == "PASS" else 1)
    elif a.command == "acceptance":
        import runpy
        from sellbird.paths import PROJECT_ROOT
        script = PROJECT_ROOT / "scripts" / "offline_acceptance.py"
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
    elif a.command == "queue":
        from sellbird.local_sales.api import get_ctx
        from sellbird.local_sales.workflows import process_queue
        print(json.dumps(process_queue(get_ctx()), indent=2))
    else:
        import uvicorn
        from sellbird.local_sales.api import create_app
        uvicorn.run(create_app(), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
