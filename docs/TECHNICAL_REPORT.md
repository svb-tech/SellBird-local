# Technical report — Local Multi-Product AI Sales Agent

## 1. Architecture

```
 Browser UI (sellbird/web/local.html, vanilla JS, no CDN)           CLI: python -m sellbird.local_sales
        │  HTTP (loopback)                                            │
        ▼                                                             ▼
 FastAPI router  sellbird/local_sales/api.py  (also mounted in Sellbird's dashboard at /api/local)
        │
        ▼
 workflows.py ── run_workflow(): ok │ degraded (rules + queue) │ queued │ blocked │ failed
   │      │               │
   │      │               └─ store.py  SQLite  data/local_sales.db  (business, products, prospects, signals,
   │      │                                       problems, scores, emails, replies, task_queue)
   │      └─ embeddings.py  Sentence Transformers (local_files_only) → TF-IDF fallback (labelled)
   ▼
 llm.py  LocalLLM ──► http://127.0.0.1:11434/api/chat  (Ollama, Qwen3)      logs/inference.jsonl
   ▲ endpoint guard: loopback only (private LAN only with explicit flag)
 hardware.py  RAM/CPU/GPU detection → model + profile recommendation
 netguard.py  socket-level block of non-loopback traffic (used by the offline acceptance test)
```

Local inference (`llm.py`) is the only module that speaks to a model. External web research (Sellbird's existing
`collectors/`, Playwright/BeautifulSoup) is separate and is not required by the eight workflows.

## 2. Workflows and what is AI vs deterministic

| # | Workflow | AI (Ollama) | Deterministic (labelled) |
|---|---|---|---|
| 1 | Business profile | JSON profile from company docs | file ingestion |
| 2 | Product ICP | JSON ICP per product | numeric normalisation of employee range |
| 3 | Prospect research | structured summary of stored text | prospect file loading |
| 4 | Buying signals | type, strength 1–5, verbatim evidence | **grounding check** (≥ 80 % of evidence words must exist in source text, else score weight 0); keyword rules as outage fallback (`method: rules`) |
| 5 | Problem statement | 1–2 sentence statement from grounded evidence | refuses to generate when no grounded signal exists |
| 6 | Email | subject + body | validator (40–130 words, name/company present, no placeholders/hype, exactly one question), one repair retry, signature template appended by code |
| 7 | Score (0–100) | LLM fit 0–10 → 15 pts | ICP match 40 (industry 15, size 10, title 10, geography 5), signals 30 (top 3 grounded), embedding similarity 15; if the LLM is down the total is rescaled from the 85 rule points and flagged |
| 8 | Reply classification | 9 categories + confidence + next action | unsubscribe / out-of-office regexes (safety-critical), keyword fallback on outage |

Nothing is hard-coded as generated output: every AI field comes from a logged model call (`inference_id` is returned
in each result) and every deterministic field is labelled `method: rules` or named in the breakdown.

## 3. Hardware awareness and model selection

`hardware.detect_hardware()` reads RAM (psutil → `/proc/meminfo` → `sysctl` → `wmic`), CPU model/cores, and GPUs
(`nvidia-smi`; Apple Silicon unified memory counted as 65 % of RAM; Windows adapters listed without VRAM).
`recommend_model()`:

* budget = GPU VRAM if a GPU has ≥ 4 GB, else **50 % of RAM** (leaves room for OS, Ollama KV cache, the app);
* a model fits when `download_size × 1.3 ≤ budget`; choose the largest fitting tag from
  `qwen3:1.7b (1.4 GB) < 4b (2.5) < 8b (5.2) < 14b (9.3) < 30b (19)`;
* CPU-only caps: standard profile ≤ 4b, high-performance RAM (≥ 32 GB) ≤ 14b, because larger CPU-only models are too slow;
* profile `standard` → `num_ctx 4096, num_predict 700`; `high_performance` (≥ 32 GB RAM or ≥ 12 GB VRAM) → `8192 / 1200`.
No GPU is assumed anywhere. Unit-tested in `tests/test_local_sales.py::test_hardware_recommendation_tiers`.

## 4. Offline behaviour

After `ollama pull` (and optionally caching the embedding model) every workflow uses only loopback HTTP and local
files. `scripts/offline_acceptance.py` enforces this with `netguard` and fails if any external connection is even
attempted. Embeddings load with `local_files_only=True` / `HF_HUB_OFFLINE=1`. The UI has no CDN, fonts or analytics.

## 5. Failure handling

See `LOCAL_AI_README.md` (table). Key property: `LocalAIUnavailable` is the only error that triggers queue/fallback,
and no code path constructs a client for any other provider. Queue entries are de-duplicated and replayed by
`process_queue()`; replay stops at the first connectivity error so nothing is lost.

## 6. Observability

Every call appends to `logs/inference.jsonl`: id, timestamp, workflow, endpoint, model requested/served, prompt and
completion tokens, tokens/sec, latency, `cost_usd`, success/error. The UI shows the last calls; `/api/local/status`
reports `all_logged_calls_local` and `total_cost_usd`.

## 7. Major decisions

* **Own package and DB** instead of editing Sellbird's 78 KB `state.py`: small reviewable diff, no migration risk.
* **JSON-schema constrained decoding** (`format` = schema) so a 1.7 B model returns parseable output; plus bounded repair retries.
* **Grounding check** because small models hallucinate evidence; ungrounded signals are stored but score 0 and are flagged.
* **Rules before LLM for safety-critical replies** (unsubscribe/OOO) so opt-outs never depend on a model.
* **Emails are drafts only.** Nothing is ever sent by the local agent.
* **Synchronous httpx + FastAPI thread pool** — simplest correct concurrency model for one local Ollama instance.

## 8. Test evidence

* `pytest tests/test_local_sales.py` — 9 tests (hardware tiers, non-local endpoint refusal, end-to-end pipeline + inference-log assertions,
  grounding, email validation, no-signal → no email, deterministic reply rules, Ollama-down queue/fallback, missing prerequisites).
* The full Sellbird suite plus the new local tests pass (3714 passed, 29 skipped).
* `evidence/offline_acceptance_*.{json,md}` — **generated by you on your machine with the real model** (not shipped here).
