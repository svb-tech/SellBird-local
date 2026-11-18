# Licences of models, libraries and major dependencies

Verified = read from installed package metadata on the build machine. Upstream = taken from the project's published
licence; re-check the model card / repo before submission if your evaluators require it.

| Component | Role | Licence | Status |
|---|---|---|---|
| Sellbird, formerly Harvey (ethanplusai/sellbird) | base project | AGPL-3.0-only (v0.4+; Harvey ≤0.3.x was MIT) | repo `LICENSE` + `NOTICE` |
| Ollama | local inference server | MIT | upstream |
| Qwen3 (qwen3:1.7b / 4b / 8b / 14b / 30b) | LLM | Apache-2.0 | upstream (Qwen model cards) |
| sentence-transformers | local embeddings library | Apache-2.0 | upstream |
| all-MiniLM-L6-v2 | embedding model | Apache-2.0 | upstream |
| SQLite (Python `sqlite3`) | storage | Public domain | upstream |
| Python | runtime | PSF | upstream |
| FastAPI 0.163 | backend | MIT | verified |
| Uvicorn 0.54 | ASGI server | BSD-3-Clause | verified |
| Pydantic 2.14 | validation | MIT | verified |
| httpx 0.28 | HTTP client (loopback Ollama calls + existing collectors) | BSD-3-Clause | verified |
| PyYAML 6.0 | config | MIT | verified |
| python-dotenv 1.2 | env loading | BSD-3-Clause | verified |
| Playwright 1.63 | public-web research (existing Sellbird collectors) | Apache-2.0 | verified |
| beautifulsoup4 4.15 | HTML parsing (existing Sellbird collectors) | MIT | verified |
| dnspython 2.9 | existing Sellbird e-mail finder | ISC | verified |
| aiosmtplib 5.1 / aiosqlite 0.22 / pytz | existing Sellbird modules | MIT | verified |
| psutil (optional) | RAM detection | BSD-3-Clause | upstream |
| pytest / pytest-asyncio (dev) | tests | MIT / Apache-2.0 | verified |
| Scrapy | named in the brief | BSD-3-Clause | **not used** (see KNOWN_LIMITATIONS) |

All model weights used are Apache-2.0 (open-weight *and* permissive). No non-commercial or "community" licence models are used.


**Note on the combined work:** the base project is AGPL-3.0-only, so this extension (`sellbird/local_sales/`, scripts, tests) is distributed under the same licence. If you run it as a network service for other people you must offer them the source.
