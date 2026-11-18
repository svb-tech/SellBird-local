# Sellbird Local — Zero-Cost Multi-Product AI Sales Agent

Fork and extension of [Harvey AI Sales Agent](https://github.com/ethanplusai/harvey) for the **EDXSO AI Engineer Intern — Assignment 2**.

It transforms the original project into a **fully local, multi-product AI sales agent**. All language-model inference runs on your own machine through [Ollama](https://ollama.com) using **Qwen3 (Apache-2.0)**.

There are no OpenAI, Anthropic, Gemini, Codex, or other paid/hosted AI inference calls, and no cloud fallback. If Ollama is unavailable, the application displays a clear error, uses clearly labelled deterministic rules where possible, and queues remaining tasks for later processing.

---

## Project Submission

| Deliverable | Link |
|---|---|
| GitHub Repository | https://github.com/svb-tech/SellBird-local |
| Demo Video (Google Drive) | [https://drive.google.com/drive/folders/1YYlzhsOU4FnfVmDtlE1Vd48mEiJDYrKP] (PASTE_GOOGLE_DRIVE_LINK_HERE) |

---

## Documentation

- [Original Upstream README](README.md)
- [Technical Documentation](TECHNICAL_DOCUMENTATION.md)

---

## What it does

Eight local AI workflows, available in the web UI, the REST API and the offline acceptance script:

| # | Workflow | Input | Output |
|---|---|---|---|
| 1 | Business profile | `data/local/company_info/*.md` | Structured company profile |
| 2 | Product-specific ICP | Profile + one product's sales context | One ICP per product |
| 3 | Prospect research | Previously stored prospect text (`data/local/prospects.json`) | Structured research summary |
| 4 | Buying-signal classification | Research text | Signal type, strength 1–5, verbatim evidence (must be grounded in the source text) |
| 5 | Problem statement | Grounded signals | 1–2 sentence problem statement |
| 6 | Personalised email | Prospect + product + problem | Validated draft (drafts only, never sent) |
| 7 | Prospect scoring | All of the above | Score 0–100 and tier A/B/C with a full breakdown |
| 8 | Reply classification | Sample or pasted replies | One of 9 categories + confidence + next action |

Scoring: 40 points ICP rules + 30 points grounded signals + 15 points embedding similarity + 15 points LLM fit.

## Quick start

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed once. Internet is needed only for installation and the model download.

```bash
./run_local.sh          # macOS / Linux
run_local.bat           # Windows
```

The script creates a virtual environment, installs dependencies, starts Ollama, pulls the model that fits your hardware
and serves the app at **http://127.0.0.1:8765**.

Manual equivalent:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e . -r requirements-local.txt
sellbird local hardware                               # shows RAM/CPU/GPU and the recommended model
ollama pull qwen3:1.7b                                # ~1.4 GB, or the tag recommended above
sellbird local serve                                  # same as: python -m sellbird.local_sales serve
```

Optional but recommended (better semantic scoring, once while online):

```bash
python scripts/verify_local.py embeddings --download
```

Without it the app uses a pure-Python TF-IDF similarity and says so in every score (`_embedding_backend`).

## Hardware-aware model selection

The app detects RAM, CPU and GPU and recommends a model. **No GPU is required.**

| Profile | Machine | Model | Context |
|---|---|---|---|
| `standard` | Ordinary laptop, CPU only, 4–16 GB RAM | `qwen3:1.7b` (about 4 GB RAM) or `qwen3:4b` (8 GB or more) | 4096 |
| `high_performance` | 32 GB or more RAM, or GPU with 12 GB or more VRAM | `qwen3:8b` / `14b` (CPU cap 14b) / `30b` (32 GB or more VRAM) | 8192 |

Override with `SELLBIRD_LOCAL_MODEL=qwen3:4b`. Selection rules are in the technical documentation, section 4.

## Demo walkthrough (about 8 minutes)

1. `python -m sellbird.local_sales hardware`, then `python -m sellbird.local_sales serve`.
2. Open the UI and click **Load company + prospect files** (two products appear).
3. **Generate business profile**.
4. Select a product, then **Generate ICP**. Switch to the other product to show a different ICP.
5. **Run pipeline** for 4 prospects: research, signals, problem statement, score, email.
6. Open prospect details: grounded evidence, score breakdown, email and its validation checks.
7. Paste a reply and **Classify**, or run the 36 labelled sample replies.
8. Stop Ollama to show the red error banner, `queued` / `degraded` states, then **Process queue** after restart.
9. **Disconnect the network** and run the offline acceptance test (below).

The full recording script is in [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

## Offline acceptance test

1. While online, once: install, `ollama pull <model>`, optionally cache the embedding model.
2. **Disconnect Wi-Fi / Ethernet.**
3. Run:

```bash
ollama serve                                        # if not already running
python scripts/offline_acceptance.py --max-prospects 4    # or: sellbird local acceptance
python -m sellbird.local_sales audit
```

The script installs a socket guard that raises on any non-loopback connection or DNS lookup, runs all 8 workflows on a
fresh database, and writes `evidence/offline_acceptance_<timestamp>.{json,md}` with per-step PASS/FAIL, every inference
endpoint (all must be `127.0.0.1`), total cost (`$0.00`) and the number of blocked external attempts (must be 0).

## Failure handling (no silent cloud fallback)

| Situation | Behaviour |
|---|---|
| Ollama not running or model missing | Red banner with the exact fix (`ollama serve`, `ollama pull ...`) |
| Profile, ICP, research, problem, email | Status `queued`, stored in the SQLite `task_queue`; replayed by **Process queue** or `python -m sellbird.local_sales queue` |
| Signals, scoring, reply classification | Status `degraded`: labelled rules (`method: rules`) run now and the LLM re-run is queued |
| `OLLAMA_HOST` points to a non-local host | Refused at start-up (`NonLocalEndpointError`) |
| Model returns invalid JSON | Up to 2 repair retries, then `status: failed` with the reason |

## Proof of zero cost

| Check | How |
|---|---|
| No hosted-LLM code in the local agent | `python -m sellbird.local_sales audit` scans `sellbird/local_sales/*.py` for hosted-LLM hostnames and SDK imports and checks paid-AI environment variables |
| Every call is local and free | `logs/inference.jsonl`: one line per call with endpoint, model, tokens, tokens/sec, latency and `cost_usd` |
| No outside traffic while offline | `scripts/offline_acceptance.py` socket guard, `logs/network_attempts.jsonl` |

Details and limits of this claim: [`docs/COST_AND_DEPENDENCY_STATEMENT.md`](docs/COST_AND_DEPENDENCY_STATEMENT.md).

## Project layout

```
sellbird/local_sales/      New code: the local sales agent
  llm.py                   Ollama client, loopback-only guard, inference log
  workflows.py             The 8 workflows, pipeline, queue replay
  hardware.py              RAM/CPU/GPU detection and model recommendation
  embeddings.py            Sentence Transformers (offline) with labelled TF-IDF fallback
  store.py                 SQLite storage (data/local_sales.db)
  api.py                   FastAPI router (/api/local/*) and UI route
  netguard.py              Socket-level block of external traffic (acceptance test)
  brain_transport.py       Ollama transport for Sellbird's own "brain"
sellbird/web/local.html    Dashboard UI (vanilla JS, no CDN)
data/local/                Demo company files, 12 prospects, 36 labelled replies (synthetic)
scripts/                   offline_acceptance.py, verify_local.py
tests/                     test_local_sales.py, test_ollama_brain_provider.py (+ upstream tests)
docs/                      Technical documentation, licences, cost statement, limitations, demo script
```

Changes to existing upstream files are small: `dashboard.py` (mounts `/api/local`), `config.py` and `brain.py`
(`ollama` added as a third `brain.provider`), `cli.py` (`sellbird local ...`), `setup.py`, `sellbird.yaml`
(`brain.provider: ollama`), `pyproject.toml` (optional `local` extras).

## Tests

```bash
pip install -e ".[dev,local]"
pytest tests/test_local_sales.py tests/test_ollama_brain_provider.py
```

These tests use `tests/fake_ollama.py`, a **test-only** stub server. The application itself never uses it.

## Documentation index

| Document | Contents |
|---|---|
| [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) | Architecture, model setup, hardware logic, offline behaviour, failure handling, decisions |
| [`docs/LICENSES.md`](docs/LICENSES.md) | Licence of every model, library and major dependency |
| [`docs/COST_AND_DEPENDENCY_STATEMENT.md`](docs/COST_AND_DEPENDENCY_STATEMENT.md) | Zero-cost evidence and disclosed limits |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | Incomplete, simulated or internet-dependent parts |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | Screen-recording script |

## Known limitations (summary)

* Prospect data, company files and sample replies are **synthetic demo data**. The loaders accept real JSON/CSV in the same shape.
* Fresh web discovery is **not** wired into the eight local workflows; they read stored text. Scrapy is not used.
* The UI is a vanilla-JS page served by FastAPI, not a React app.
* Emails are drafts only. Nothing is ever sent.
* Single-user, single-process. Local run only; no hosted URL.
* Original Sellbird modules for Claude/Codex and paid integrations remain in the repository but are never imported by the local agent.
* Small models (`qwen3:1.7b`) make mistakes. Grounding checks and validators reduce, but do not remove, them.

See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) for the full list.

## Licence

Base project: AGPL-3.0-only (see `LICENSE` and `NOTICE`). This extension is distributed under the same licence.
Model and dependency licences: [`docs/LICENSES.md`](docs/LICENSES.md).
