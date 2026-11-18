# Technical Documentation — Local Multi-Product AI Sales Agent

EDXSO AI Engineer Intern — Assignment 2 (fork of Harvey / Sellbird)

## Contents

1. Overview and scope
2. Architecture
3. Local model setup
4. Hardware detection and model selection
5. The eight workflows
6. Prospect scoring
7. Data model
8. REST API and CLI
9. Offline behaviour
10. Failure handling
11. Observability and zero-cost evidence
12. Models, libraries and licences
13. Secrets and security
14. Major implementation decisions
15. Testing
16. Mocked or simulated functionality
17. Known limitations

---

## 1. Overview and scope

The project extends the open-source Harvey (now Sellbird) outreach platform with a self-contained package,
`sellbird/local_sales/`, that implements a multi-product sales agent using **only local inference**.

* **Inference:** Ollama serving Qwen3 (Apache-2.0) on `127.0.0.1:11434`.
* **Storage:** SQLite (`data/local_sales.db`).
* **Backend:** Python, FastAPI. **Frontend:** a single vanilla-JS page (`sellbird/web/local.html`), no CDN, fonts or analytics.
* **Embeddings:** Sentence Transformers `all-MiniLM-L6-v2`, loaded offline, with a labelled TF-IDF fallback.
* **Separation:** local inference (`llm.py`) is the only module that talks to a model. External web research (Sellbird's
  existing `collectors/`, using httpx, BeautifulSoup and Playwright) is a separate, optional part of the repository and is
  not needed by the eight local workflows.

The upstream Sellbird code (Claude/Codex brain paths, Instantly, Hunter, Serper, DataForSEO, mail sending) is still in the
repository but is **never imported by `sellbird/local_sales/`**. See sections 11 and 17.

## 2. Architecture

```
 Browser UI  sellbird/web/local.html (vanilla JS)          CLI: python -m sellbird.local_sales
        |  HTTP (loopback)                                         |
        v                                                          v
 FastAPI router  sellbird/local_sales/api.py   (also mounted in Sellbird's dashboard at /api/local)
        |
        v
 workflows.py   run_workflow(): ok | degraded (rules + queue) | queued | blocked | failed
   |        |
   |        +--> embeddings.py   Sentence Transformers (local_files_only) -> TF-IDF fallback (labelled)
   |        +--> store.py        SQLite  data/local_sales.db
   v
 llm.py  LocalLLM  --->  http://127.0.0.1:11434/api/chat  (Ollama, Qwen3)      logs/inference.jsonl
   ^ endpoint guard: loopback only (private LAN only with an explicit flag)
 hardware.py   RAM / CPU / GPU detection -> model and profile recommendation
 netguard.py   socket-level block of non-loopback traffic (offline acceptance test)
```

| Module | Responsibility |
|---|---|
| `llm.py` | Ollama client. Validates the endpoint, sends JSON-schema constrained requests, retries invalid JSON, writes the inference log. Raises `LocalAIUnavailable` and never falls back. |
| `workflows.py` | The eight workflows, `run_workflow` (status handling), `run_pipeline`, `process_queue`, reply-classifier evaluation. |
| `hardware.py` | `detect_hardware()` and `recommend_model()`. |
| `embeddings.py` | `Embedder.similarity()` with backend name reported in every result. |
| `store.py` | SQLite schema and queries. |
| `api.py` | FastAPI router and UI route. |
| `netguard.py` | Patches `socket.connect` and `socket.getaddrinfo` to block and log external traffic. |
| `brain_transport.py` | Routes Sellbird's own `brain.provider: ollama` calls to the same local endpoint with the same logging and guard. |
| `__main__.py` | CLI: `hardware`, `serve`, `audit`, `queue`, `acceptance`. |

## 3. Local model setup

1. Install Ollama (MIT) from https://ollama.com. This is a one-time step that needs internet.
2. Install the Python dependencies: `pip install -e . -r requirements-local.txt`.
3. Pull the recommended model: `ollama pull qwen3:1.7b` (about 1.4 GB), or the tag printed by `sellbird local hardware`.
4. Optional, once while online: cache the embedding model with `python scripts/verify_local.py embeddings --download`.
5. Start the app: `sellbird local serve`, then open http://127.0.0.1:8765.

`run_local.sh` and `run_local.bat` perform steps 2 to 5 automatically, including the hardware-based model choice.

Model selection order: `SELLBIRD_LOCAL_MODEL` (or legacy `HARVEY_LOCAL_MODEL`), then the hardware recommendation, with
`qwen3:1.7b` as the default in `LocalLLM`.

Default generation options: `temperature 0.2`, `seed 7`, `num_ctx` and `num_predict` from the active hardware profile,
request timeout 180 s.

## 4. Hardware detection and model selection

`detect_hardware()` reads, with no network access:

* **RAM:** `psutil`, then `/proc/meminfo`, then `sysctl hw.memsize` (macOS), then `wmic` (Windows).
* **CPU:** model name and core count.
* **GPU:** `nvidia-smi` for NVIDIA (name and VRAM). Apple Silicon is detected as unified memory, with 65 % of RAM counted
  as usable by the GPU. On Windows, adapters are listed without VRAM (not reliably reported).

`recommend_model()` rules:

* **Budget:** GPU VRAM if a GPU has at least 4 GB, otherwise **50 % of RAM** (headroom for the OS, the KV cache and the app).
* **Fit:** a model fits when `download_size x 1.3 <= budget`. The largest fitting tag is chosen from the ladder.
* **CPU-only caps:** standard profile at most `qwen3:4b`; high-performance RAM class at most `qwen3:14b`, because larger
  CPU-only models are too slow for interactive use.
* **No GPU is ever assumed.** `gpu_required` is always `false`.

| Ladder tag | Approx. download | Notes |
|---|---|---|
| `qwen3:1.7b` | 1.4 GB | Default for small laptops |
| `qwen3:4b` | 2.5 GB | Recommended for 8 GB or more RAM |
| `qwen3:8b` | 5.2 GB | |
| `qwen3:14b` | 9.3 GB | CPU cap for the high-performance profile |
| `qwen3:30b` | 19 GB | MoE (3B active); needs about 32 GB of VRAM |

| Profile | Trigger | `num_ctx` | `num_predict` |
|---|---|---|---|
| `standard` (ordinary laptop) | default | 4096 | 700 |
| `high_performance` | RAM of 32 GB or more, or VRAM of 12 GB or more | 8192 | 1200 |

Example output on a 3.9 GB, 1-core, CPU-only machine: profile `standard`, recommended model `qwen3:1.7b`
(budget 1.9 GB, cap `qwen3:4b`).

## 5. The eight workflows

Every AI field comes from a logged model call (the `inference_id` is returned in each result). Every deterministic field
is labelled (`method: rules`) or named in the score breakdown.

| # | Workflow | AI part (Ollama) | Deterministic part |
|---|---|---|---|
| 1 | Business profile | JSON profile from company documents | File ingestion from `data/local/company_info/` |
| 2 | Product ICP | JSON ICP per product | Numeric normalisation of the employee range |
| 3 | Prospect research | Structured summary of stored prospect text | Prospect file loading (`prospects.json`, optional `research/<slug>.txt`) |
| 4 | Buying signals | Type, strength 1–5, verbatim evidence | **Grounding check:** at least 80 % of the evidence words must appear in the source text, otherwise the signal is stored but its score weight is 0. Keyword rules are used as the outage fallback. |
| 5 | Problem statement | 1–2 sentence statement from grounded evidence | Refuses to generate when no grounded signal exists |
| 6 | Personalised email | Subject and body | Validator: 40–130 words, prospect name and company present, no placeholders, no hype words, exactly one question. One repair retry. Signature appended by code. |
| 7 | Prospect scoring | LLM fit rating (0–10, worth 15 points) | ICP match, signals, embedding similarity (section 6) |
| 8 | Reply classification | 9 categories, confidence, next action | `unsubscribe` and `out_of_office` regexes (safety critical); keyword fallback during outages |

Reply categories: `interested`, `meeting_request`, `question`, `objection_price`, `objection_timing`, `not_interested`,
`referral`, `out_of_office`, `unsubscribe`.

**Pipeline** (`run_pipeline`): for each prospect (up to `max_prospects`) run research, signals, problem statement and
score; then rank by total score and generate emails only for tiers A and B. A prospect with no grounded signal gets no
email.

**Reply evaluation** (`evaluate_reply_classifier`): runs the 36 labelled replies in `data/local/sample_replies.json`
(4 per category) and reports accuracy, per-class results, a confusion matrix and the misclassified items. The few-shot
examples in the prompt are different texts, and a test enforces that none of the evaluation texts appear in the prompt.

## 6. Prospect scoring

Total of 100 points, then a tier:

| Component | Points | Details |
|---|---|---|
| ICP rule match | 40 | industry 15, company size 10, title 10, geography 5 |
| Buying signals | 30 | top 3 **grounded** signals: `sum(strength) / 15 x 30` (strength 1–5 each) |
| Embedding similarity | 15 | prospect text vs. product ICP; backend named in `_embedding_backend` |
| LLM fit | 15 | model fit rating 0–10 scaled to 15 |

If the LLM is unavailable, the total is rescaled from the 85 rule-based points and flagged as `degraded`.
The result includes the full breakdown so every score is explainable. Tiers: **A** is 70 or more, **B** is 50 to 69, **C** is below 50. Company size scores 10 inside the ICP range, 4 within 50 % outside it, otherwise 0. Geography scores 5 when the ICP lists no geography.

## 7. Data model

SQLite file `data/local_sales.db`:

| Table | Content |
|---|---|
| `business` | Single row: company name, raw ingested info, generated profile |
| `products` | One row per product, with product context and generated ICP |
| `prospects` | Company, contact, title, industry, size, location, source, raw notes, research |
| `signals` | Per prospect and product: type, strength, evidence, grounded flag, method, model |
| `problems` | Problem statements |
| `scores` | Total, tier and component breakdown per prospect and product |
| `emails` | Subject, body, validation checks, model |
| `replies` | Text, predicted category, confidence, method, expected label (for evaluation) |
| `task_queue` | Queued workflows: payload, status, attempts, error |

## 8. REST API and CLI

Standalone server (loopback only, no token): `http://127.0.0.1:8765`. The same router is also mounted at `/api/local`
inside Sellbird's dashboard, where it requires the dashboard token.

| Method and path | Purpose |
|---|---|
| `GET /status` | Ollama health, model, hardware profile, embedding backend, `all_logged_calls_local`, `total_cost_usd` |
| `POST /setup/load` | Load company files and prospects |
| `POST /business/profile`, `GET /business` | Workflow 1 and read |
| `GET /products`, `POST /products/{pid}/icp` | Workflow 2 |
| `GET /prospects`, `GET /prospects/{pid}` | List and details |
| `POST /run/{workflow}` | Run one workflow |
| `POST /pipeline` | Run the end-to-end pipeline |
| `GET /ranking` | Ranked prospects |
| `POST /replies/evaluate` | Run the labelled reply evaluation |
| `GET /inference-log` | Recent inference records |
| `GET /queue`, `POST /queue/process` | Inspect and replay queued tasks |
| `GET /ui` | The dashboard page |

CLI: `python -m sellbird.local_sales <hardware | serve | audit | queue | acceptance>` (also `sellbird local ...`).

## 9. Offline behaviour

After installation and model download, every workflow uses only loopback HTTP and local files.

* Ollama serves the model from local disk.
* Embeddings load with `local_files_only=True` and `HF_HUB_OFFLINE=1`; if not cached, the TF-IDF fallback is used and reported.
* The UI loads no CDN, web fonts or analytics.
* Prospect data is read from local files.

**Acceptance test** (`scripts/offline_acceptance.py`): installs `netguard` before anything else, so any non-loopback
connection or DNS lookup raises and is logged to `logs/network_attempts.jsonl`. It then runs all 8 workflows on a fresh
database and writes `evidence/offline_acceptance_<timestamp>.{json,md}` containing per-step PASS/FAIL, all inference
endpoints (must be `127.0.0.1`), total cost (`$0.00`) and the blocked-attempt count (must be 0). Exit code 0 only if every
workflow succeeded on local inference. `netguard` is a proof aid; the demo should also physically disconnect the network.

## 10. Failure handling

`LocalAIUnavailable` is the only error that triggers queueing or fallback. No code path constructs a client for any other
provider.

| Situation | Status | Behaviour |
|---|---|---|
| Ollama down or model missing | `queued` for profile, ICP, research, problem, email | Stored in `task_queue`; red banner with the exact fix |
| Ollama down | `degraded` for signals, scoring, reply classification | Labelled rules (`method: rules`) run immediately; LLM re-run is queued |
| Missing prerequisite (for example no ICP yet) | `blocked` | Clear message naming the missing step |
| `OLLAMA_HOST` not local | Refused at construction | `NonLocalEndpointError`; private LAN addresses only with `LOCAL_AI_ALLOW_PRIVATE_HOST=1` |
| Invalid JSON from the model | up to 2 repair retries, then `failed` | Reason stored and returned |

`process_queue()` re-runs queued tasks once Ollama is healthy, and stops at the first connectivity error so nothing is lost.
Queue entries are de-duplicated.

## 11. Observability and zero-cost evidence

Every model call appends one JSON line to `logs/inference.jsonl`: id, timestamp, workflow, endpoint, model requested and
served, prompt and completion tokens, tokens per second, latency, `cost_usd` (always 0.0), success or error.
The UI shows the latest calls, and `/api/local/status` reports `all_logged_calls_local` and `total_cost_usd`.

`python -m sellbird.local_sales audit` checks `sellbird/local_sales/*.py` for hosted-LLM hostnames (OpenAI, Anthropic,
Google Gemini, Cohere, Mistral, Groq, OpenRouter, Bedrock, Together, Perplexity) and SDK imports, checks that no paid-AI
API key environment variables are set, and lists the distinct endpoints seen in the inference log. It exits non-zero on failure.

**Honest scope of the claim:** the audit covers the local agent package. The upstream Sellbird modules for Claude/Codex and
paid services remain in the repository, are not imported by the local agent, and are not used by any local workflow. In
addition, `sellbird.yaml` sets `brain.provider: ollama` so Sellbird's own pipeline also uses local inference by default
(the upstream code default is `claude`, so this block must stay in place). Do not run Sellbird's legacy autonomous loop
with paid keys configured if a zero-cost run is required.

## 12. Models, libraries and licences

| Component | Role | Licence |
|---|---|---|
| Qwen3 (`1.7b`, `4b`, `8b`, `14b`, `30b`) | LLM | Apache-2.0 |
| Ollama | Local inference server | MIT |
| sentence-transformers | Embeddings library | Apache-2.0 |
| all-MiniLM-L6-v2 | Embedding model | Apache-2.0 |
| SQLite | Storage | Public domain |
| FastAPI | Backend | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| Pydantic | Validation | MIT |
| httpx | HTTP client | BSD-3-Clause |
| PyYAML | Config | MIT |
| python-dotenv | Env loading | BSD-3-Clause |
| Playwright | Public-web research (existing collectors) | Apache-2.0 |
| beautifulsoup4 | HTML parsing (existing collectors) | MIT |
| psutil | RAM detection | BSD-3-Clause |
| Sellbird / Harvey (base project) | Base application | AGPL-3.0-only (v0.4 and later) |

The complete table, including which entries were verified from installed package metadata and which are taken from
upstream, is in `docs/LICENSES.md`. Because the base project is AGPL-3.0-only, this extension is distributed under the same
licence. Scrapy is named in the brief but is not used (see section 17).

## 13. Secrets and security

* The local agent needs **no credentials** and reads no API keys.
* No secrets are stored in source. Upstream secrets handling (`.env`, keyring) is unchanged and unused by the local agent.
* The standalone server binds to `127.0.0.1` only. Inside the dashboard the routes sit behind the dashboard token
  (covered by a test).
* Evidence returned by the model must exist in the stored source text (grounding check), which limits the effect of invented or injected content on scores.

## 14. Major implementation decisions

* **Own package and database** instead of editing Sellbird's large `state.py`: small, reviewable diff and no migration risk.
* **JSON-schema constrained decoding** (Ollama `format`) so a 1.7B model returns parseable output, plus bounded repair retries.
* **Grounding check** because small models invent evidence: ungrounded signals are stored but score 0 and are flagged.
* **Rules before LLM for safety-critical replies:** unsubscribe and out-of-office never depend on a model.
* **Drafts only:** the local agent never sends email.
* **Labelled fallbacks:** every non-LLM result carries `method: rules` or an embedding backend name, so nothing deterministic is presented as AI output.
* **Synchronous httpx with the FastAPI thread pool:** simplest correct model for a single local Ollama instance.
* **Loopback-only endpoint guard** in the client, so a misconfigured `OLLAMA_HOST` can never route to a hosted service.

## 15. Testing

```bash
pip install -e ".[dev,local]"
pytest tests/test_local_sales.py tests/test_ollama_brain_provider.py
```

Verified when preparing this document: **16 passed** (11 tests in `test_local_sales.py` and 5 in
`test_ollama_brain_provider.py`), `python -m sellbird.local_sales audit` returned **PASS**, and the existing config tests
still pass with the `brain` block in `sellbird.yaml` (37 passed across the four config and local test files).

Covered: hardware tiers, refusal of non-local endpoints, end-to-end pipeline with inference-log assertions, grounding,
email validation and signature, no signal means no email, deterministic unsubscribe and out-of-office rules, Ollama down
(queue and labelled fallback), missing prerequisites, reply evaluation set balance and prompt leakage, confusion matrix,
the Ollama brain provider and dashboard token protection.

These tests run against `tests/fake_ollama.py`, a **test-only** stub. They prove the plumbing, not the quality of a real
model. Real evidence must be produced with the real model (see below).

**Evidence to generate on the submission machine** (not included in the repository):

* `evidence/offline_acceptance_*.json` and `.md` from `scripts/offline_acceptance.py` with the network disconnected.
* `logs/inference.jsonl` from a real run.
* `python scripts/verify_local.py persistence --live` and `embeddings` outputs.
* Screenshots and the screen recording.

## 16. Mocked or simulated functionality

* `data/local/prospects.json` (12 prospects), the company files and `data/local/sample_replies.json` (36 labelled replies)
  are **synthetic demo data**. The "previously downloaded" prospect pages are simulated text.
* `tests/fake_ollama.py` is a test stub and is never used by the application.
* Nothing in the eight workflows returns hard-coded output: AI fields come from logged Ollama calls.

## 17. Known limitations

1. Real-model output quality has not been measured in the build environment (no Ollama there). A 1.7B model will make
   mistakes; use `qwen3:4b` or larger if RAM allows. Report the accuracy measured on your machine with the model tag.
2. Fresh web discovery is not wired into the eight local workflows. Sellbird's collectors remain in the repository.
   Scrapy is not used (the brief allows an equivalent implementation).
3. The frontend is vanilla JS served by FastAPI, not React.
4. Legacy paid integrations and Claude/Codex brain paths still exist in the repository, unused by the local agent.
5. If `sentence-transformers` or its cached model is missing, TF-IDF similarity is used (reported as such), so scores are less semantic.
6. CPU-only laptops take seconds to tens of seconds per call; the 4-prospect pipeline takes several minutes.
7. Emails are drafts only. There is no mail integration in the local path.
8. Single-user, single-process. Local run only; no hosted URL.
9. The keyword fallback used while Ollama is down is weak (about 39 % on the 36 sample replies) and is marked "Review manually".

The full list is in `docs/KNOWN_LIMITATIONS.md`.
