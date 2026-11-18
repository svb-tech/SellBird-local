# Cost and dependency statement

**Zero LLM cost.** The local sales agent (`sellbird/local_sales/`) uses no paid or hosted AI.

| Item | Status |
|---|---|
| OpenAI / Anthropic / Gemini / Codex / Claude Code | not used, not imported, no keys read |
| Hosted inference or per-token billing | none; every inference record has `cost_usd: 0.0` |
| Inference endpoint | `POST http://127.0.0.1:11434/api/chat` (Ollama) only |
| Non-loopback `OLLAMA_HOST` | rejected at construction (`assert_local_endpoint`) |
| Fallback when Ollama fails | none to cloud; error + deterministic rules + queue |
| Paid prospecting/enrichment APIs | none used. Prospects come from a local file (`data/local/prospects.json`) |
| Secrets in source | none; the local agent needs no credentials |
| Embeddings | Sentence Transformers loaded with `local_files_only=True` (or TF-IDF fallback) |

## How to verify (reproducible)

1. `python -m sellbird.local_sales audit` — scans `sellbird/local_sales/*.py` for hosted-LLM hostnames/SDK imports and checks that no paid-AI environment variables are set. Expected: `PASS`.
2. `logs/inference.jsonl` — one JSON line per call: workflow, endpoint, model served, token counts, tokens/sec, latency, `cost_usd`.
3. `python scripts/offline_acceptance.py` — blocks every non-loopback socket/DNS call and reports how many attempts were made (expected 0).
4. Physically disconnect the network and repeat step 3.

## Disclosed limitations of this claim

* The **original Sellbird modules** (`harvey/brain.py` Claude/Codex paths, `integrations/quota.py`, Instantly/Hunter/Serper/DataForSEO integrations, `harvey run` heartbeat) still exist in the repository. They are **not imported or called** by the local agent, and `sellbird.yaml` sets `brain.provider: ollama` (the upstream code default is `claude`, so keep that block in place). They were not deleted to keep the fork diff reviewable. Do not run Sellbird's legacy autonomous loop with paid keys configured if you want a zero-cost run.
* Initial installation and model download need the internet (`pip`, `ollama pull`, optional HuggingFace cache).
* Fresh prospect discovery via the public web needs the internet; the demo uses previously stored data.
