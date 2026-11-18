# Sellbird — Local Zero-Cost Multi-Product AI Sales Agent

Fork/extension of [ethanplusai/sellbird](https://github.com/ethanplusai/sellbird) for **EDXSO AI Engineer Intern — Assignment 2**.
All language-model inference runs on **local Ollama** (default `qwen3:1.7b`, Apache-2.0). No OpenAI / Anthropic / Gemini /
Codex / any paid API is called, and there is **no cloud fallback**: if Ollama is down the app shows an error, runs
labelled deterministic rules where possible, and queues the rest.

New code lives in `sellbird/local_sales/`. Changes to existing Sellbird files are small and listed here:
`dashboard.py` (mounts the `/api/local` router), `config.py` / `brain.py` (`ollama` added as a third `brain.provider`),
`cli.py` (`sellbird local ...`, `--provider ollama`), `setup.py` (wizard offers Ollama), `pyproject.toml` (optional
`local` extras), `.gitignore`, `.env.example`.

## Quick start (one command)

Install [Ollama](https://ollama.com) once (free), then:

```bash
./run_local.sh        # macOS / Linux
run_local.bat         # Windows
```

It creates the virtualenv, installs everything, starts Ollama, downloads the model that fits your RAM/GPU
(the only step that needs internet) and serves the app at **http://127.0.0.1:8765**. After that it works offline.

Manual equivalent:

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e . -r requirements-local.txt
ollama pull qwen3:1.7b                                    # ~1.4 GB; or whatever `sellbird local hardware` recommends
sellbird local hardware                                   # RAM/CPU/GPU + recommended model
sellbird local serve                                      # = python -m sellbird.local_sales serve
```

`sellbird local <hardware|serve|audit|queue|acceptance>` is the same CLI as `python -m sellbird.local_sales`.

## Claude, Codex and Ollama — one app, three brains

Sellbird's own pipeline (`sellbird setup` / `sellbird run`) picks its "brain" in `sellbird.yaml`:

```yaml
brain:
  provider: ollama      # claude (default, your Claude plan) | codex (your ChatGPT plan) | ollama (local, free)
  model: qwen3:4b       # ollama: any tag you pulled; empty = chosen for your hardware
```

* `ollama` — all calls go to your own Ollama (`sellbird/local_sales/brain_transport.py`). Non-local endpoints are
  refused, calls are logged to `logs/inference.jsonl` with cost `$0.00`, and if Ollama is down the call returns an
  error/empty result and **never falls back to Claude, Codex or any cloud API**.
* `claude` / `codex` — unchanged upstream behaviour (opt-in, uses your own subscription via their CLIs). They are
  **never used by `sellbird/local_sales/`**, the zero-cost sales agent in this document, which is Ollama-only.
* No automatic switching between the three. `sellbird status --check` reports the active one.

**The sales-agent UI** can also run inside the main dashboard: `sellbird dashboard`, take the printed unlock link and
change its path to `/static/local.html` (keep the `#token=...` part). Its API (`/api/local/*`) needs that token;
the standalone `serve` app is loopback-only and needs none.

Optional (better semantic scoring): cache the embedding model once while online:
`python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"`.
Without it the app uses a pure-Python TF-IDF similarity and says so in every score (`_embedding_backend`).

## Verification checks (write proof to `evidence/`)

```bash
python scripts/verify_local.py persistence            # SQLite: write -> close -> reopen in a NEW process, hashes must match
python scripts/verify_local.py persistence --live     # after a real run: fresh process reads data/local_sales.db
python scripts/verify_local.py embeddings --download  # once, online: cache all-MiniLM-L6-v2
python scripts/verify_local.py embeddings             # offline: must report sentence-transformers, not TF-IDF
```

The UI status line and `/api/local/status` show the active embedding backend and, if it is the TF-IDF fallback, why.

## Model selection by hardware

| Profile | Machine | Model | num_ctx | Notes |
|---|---|---|---|---|
| standard | ordinary laptop, CPU only, 4–16 GB RAM | `qwen3:1.7b` (4 GB RAM) / `qwen3:4b` (>= 8 GB) | 4096 | no GPU needed |
| high_performance | >= 32 GB RAM or GPU >= 12 GB VRAM | `qwen3:8b` / `14b` (CPU cap 14b) / `30b` (>= 32 GB VRAM) | 8192 | |

Override: `SELLBIRD_LOCAL_MODEL=qwen3:4b` (`HARVEY_LOCAL_MODEL` still works). Full logic: `docs/TECHNICAL_REPORT.md` §3.

## The eight workflows (all in the UI, API and `scripts/offline_acceptance.py`)

1. Business profile ← `data/local/company_info/*.md`
2. Product-specific ICP (one per product)
3. Prospect research ← previously downloaded text in `data/local/prospects.json` (+ optional `data/local/research/<slug>.txt`)
4. Buying-signal classification (evidence must be grounded in the source text or it scores 0)
5. Problem statement
6. Personalised email (validated: length, name, no placeholders, no hype words, exactly one question) — drafts only, never auto-sent
7. Prospect scoring: 40 rules (ICP match) + 30 signals + 15 embedding similarity + 15 LLM fit = 100, tiers A/B/C
8. Reply classification (9 categories; unsubscribe / out-of-office are deterministic)

## Offline acceptance test

1. Online, once: install, `ollama pull qwen3:1.7b`, (optional) cache the embedding model.
2. **Disconnect Wi-Fi / unplug Ethernet.**
3. `ollama serve` (if not already running), then:
   ```bash
   python scripts/offline_acceptance.py --max-prospects 4   # or: sellbird local acceptance
   sellbird local audit
   ```
   The script installs a socket guard that **raises on any non-loopback connection or DNS lookup** and logs it, runs all 8
   workflows on a fresh DB and writes `evidence/offline_acceptance_<time>.{json,md}` containing: per-step PASS/FAIL,
   every inference endpoint (must all be `127.0.0.1`), total cost (`$0.00`), blocked external attempts (must be 0).
4. Open the UI (`serve`) with the network still off and show the **Local inference log** table.

## Failure handling (no silent cloud fallback)

| Situation | Behaviour |
|---|---|
| Ollama not running / model missing | Red status banner with the exact fix (`ollama serve`, `ollama pull …`) |
| research, ICP, problem, email, profile | status `queued`; stored in SQLite `task_queue`; **Process queue** button / `python -m sellbird.local_sales queue` re-runs when Ollama is back |
| signals, scoring, reply classification | status `degraded`: labelled rules (`method: rules`) run immediately **and** the LLM re-run is queued |
| `OLLAMA_HOST` points to a public host | refused at start-up (`NonLocalEndpointError`) |
| model returns invalid JSON | up to 2 repair retries, then `status: failed` with the reason |

## Tests

```bash
pytest tests/test_local_sales.py tests/test_ollama_brain_provider.py       # uses tests/fake_ollama.py (TEST-ONLY stub, never used by the app)
```

See also: `docs/TECHNICAL_REPORT.md`, `docs/LICENSES.md`, `docs/COST_AND_DEPENDENCY_STATEMENT.md`,
`docs/KNOWN_LIMITATIONS.md`, `docs/DEMO_SCRIPT.md`.
