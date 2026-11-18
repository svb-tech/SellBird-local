# Known limitations (read before submitting)

1. **Real-model output quality is unverified by the author of this change set.** The code was developed in an environment
   without Ollama; plumbing was tested against `tests/fake_ollama.py` (TEST-ONLY stub). All real evidence (acceptance report,
   inference log, screenshots, reply-classification accuracy) must be produced on your machine with `qwen3:1.7b`.
   A 1.7 B model will make mistakes (weaker signal strength judgement, occasional off-tone emails). The validators,
   grounding check and repair retry reduce but do not remove this. Use `qwen3:4b` if your RAM allows.
2. **Prospect data is SYNTHETIC.** `data/local/prospects.json`, company files and sample replies are fictional demo data
   ("previously downloaded pages" are simulated text). The loader accepts real JSON/CSV in the same shape.
3. **Fresh web discovery is not wired into the new workflows.** Sellbird's existing collectors (httpx + BeautifulSoup +
   Playwright) remain in the repo, but the eight local workflows read stored text. **Scrapy is not used** (the brief permits an equivalent implementation).
4. **Frontend:** a vanilla-JS page served by FastAPI (also reachable inside Sellbird's dashboard) — not a React app.
5. **Legacy Sellbird paid integrations still exist in the repo** (Claude/Codex brain paths, Instantly, Hunter, Serper,
   DataForSEO). They are not used by the local agent; see the cost statement. Remove them if your reviewers want a stricter diff.
6. **Embeddings:** if `sentence-transformers` or its cached model is absent the TF-IDF fallback is used (reported as such);
   scores are then less semantic.
7. **Performance:** on a CPU-only laptop each call takes seconds to tens of seconds; the full pipeline for 4 prospects
   is several minutes. Use `--max-prospects` and the per-prospect UI.
8. **No sending.** E-mails are drafts; there is no mail integration in the local path.
9. **Single-user, single-process** (module-level context). Not hardened for multi-tenant deployment.
10. **Deployment:** local-run only; no hosted URL (a hosted version would need self-hosted GPU/CPU inference).

## Reply classification evaluation

* The 36 labelled replies in `data/local/sample_replies.json` (4 per category) are **synthetic**, written for this project. The
  prompt's few-shot examples are different texts (a test enforces that none of the eval texts appear in the prompt).
* Accuracy depends on the model: small models (qwen3:1.7b) confuse neighbouring categories (referral / not_interested,
  objection_price / question). Report the figure measured on your machine, with the model tag, from
  `evidence/offline_acceptance_*.json` (`per_class_correct`, `confusion`, `misclassified`).
* One original sample (MediCore Labs, "Let's book a demo next week") was relabelled `interested` → `meeting_request`
  because the project's own category definition says a demo request is `meeting_request`.
* `unsubscribe` and `out_of_office` are handled by deterministic rules (safety-critical, labelled `method: rules`).
* The keyword fallback used only while Ollama is down scores about 39% on the 36 replies and is marked "Review manually".
