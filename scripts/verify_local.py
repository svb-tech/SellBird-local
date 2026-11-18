#!/usr/bin/env python
"""Verification checks for the local sales agent. Each writes evidence/verify_<check>.json.

  python scripts/verify_local.py persistence            # SQLite survives a process restart
  python scripts/verify_local.py persistence --live     # read data/local_sales.db left by a real run
  python scripts/verify_local.py embeddings --download  # cache the model ONCE (needs internet)
  python scripts/verify_local.py embeddings             # offline: must use Sentence Transformers

Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "evidence"
TABLES = ["business", "products", "prospects", "signals", "problems", "scores", "emails", "replies", "task_queue"]


def _write(name: str, result: dict) -> None:
    EVIDENCE.mkdir(exist_ok=True)
    result["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (EVIDENCE / f"verify_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nRESULT: {result['result']}  (evidence/verify_{name}.json)")


# --------------------------------------------------------------------- persistence
READBACK = """
import hashlib, json, sqlite3, sys
con = sqlite3.connect(sys.argv[1]); con.row_factory = sqlite3.Row
out = {}
for t in sys.argv[2:]:
    rows = [dict(r) for r in con.execute(f"SELECT * FROM {t} ORDER BY 1")]
    for r in rows:
        r.pop("created_at", None); r.pop("updated_at", None); r.pop("done_at", None)
    out[t] = {"rows": len(rows), "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()}
out["_integrity"] = con.execute("PRAGMA integrity_check").fetchone()[0]
print(json.dumps(out))
"""


def _digest(db: Path) -> dict:
    r = subprocess.run([sys.executable, "-c", READBACK, str(db), *TABLES], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout)


def check_persistence(live: bool) -> dict:
    from sellbird.local_sales import workflows as wf
    from sellbird.local_sales.store import DB_PATH, Store

    if live:
        if not Path(DB_PATH).exists():
            return {"check": "persistence", "mode": "live", "db": str(DB_PATH), "result": "FAIL",
                    "error": "data/local_sales.db does not exist yet. Run the app / acceptance test first."}
        d = _digest(Path(DB_PATH))
        empty = [t for t in ("business", "products", "prospects", "signals", "scores") if d[t]["rows"] == 0]
        ok = d["_integrity"] == "ok" and not empty
        return {"check": "persistence", "mode": "live (fresh process reads the DB written by an earlier run)",
                "db": str(DB_PATH), "tables": d, "empty_core_tables": empty, "result": "PASS" if ok else "FAIL"}

    tmp = Path(tempfile.mkdtemp()) / "persist.db"
    s = Store(tmp)
    wf.ingest_company_files(s)
    wf.load_prospects(s)
    ctx = wf.Ctx(s, None, None)  # deterministic workflows only: no LLM needed for this check
    pid = s.products()[0]["id"]
    for pr in s.prospects():
        wf.wf_signals_rules(ctx, {"prospect_id": pr["id"], "product_id": pid})
    wf.classify_reply_rules(ctx, {"text": "Please unsubscribe me"})
    s.enqueue("research", {"prospect_id": 1}, "simulated Ollama outage")
    before = {t: len(s.q(f"SELECT 1 FROM {t}")) for t in TABLES}
    live_digest = _digest(tmp)  # read through a second connection/process while the first is still open
    s.conn.close()  # "app exit"
    after = _digest(tmp)  # brand-new process, brand-new connection
    reopened = Store(tmp)
    ok_rows = all(after[t]["rows"] == before[t] for t in TABLES)
    ok_hash = all(after[t]["sha256"] == live_digest[t]["sha256"] for t in TABLES)
    queue_ok = reopened.queue("queued")[0]["workflow"] == "research"
    ok = ok_rows and ok_hash and queue_ok and after["_integrity"] == "ok" and before["products"] > 0 and before["prospects"] > 0
    return {"check": "persistence", "mode": "write -> close -> reopen in a new process", "db": str(tmp),
            "rows_written": before, "rows_after_restart": {t: after[t]["rows"] for t in TABLES},
            "content_hashes_match": ok_hash, "queued_task_survived_restart": queue_ok,
            "integrity_check": after["_integrity"], "result": "PASS" if ok else "FAIL"}


# --------------------------------------------------------------------- embeddings
PAIRS = [  # (anchor, related, unrelated)
    ("We suffered a ransomware attack and our backups failed",
     "Need a disaster recovery plan to restore encrypted data quickly",
     "The bakery sells fresh sourdough bread every morning"),
    ("Hiring a new compliance officer ahead of the audit",
     "Looking for software to track regulatory obligations",
     "Our football team won the weekend tournament"),
]


def check_embeddings(download: bool) -> dict:
    from sellbird.local_sales import embeddings as emb

    if download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            from sentence_transformers import SentenceTransformer

            SentenceTransformer(emb.MODEL_NAME)
            print(f"cached {emb.MODEL_NAME} (this is the only step that needs internet)")
        except Exception as e:
            return {"check": "embeddings", "result": "FAIL",
                    "error": f"download failed: {type(e).__name__}: {e}. Needs internet once, and "
                             "`pip install sentence-transformers` (see requirements-local.txt)."}
        os.environ["HF_HUB_OFFLINE"] = "1"

    e = emb.Embedder()
    rows, sem_ok = [], True
    for a, rel, unrel in PAIRS:
        s_rel, s_unrel = e.similarity(a, rel), e.similarity(a, unrel)
        # TF-IDF scores ~0 on these paraphrases (no shared words); a real encoder separates them.
        rows.append({"anchor": a, "related": round(s_rel, 3), "unrelated": round(s_unrel, 3)})
        sem_ok &= s_rel > s_unrel + 0.15
    is_st = e.backend.startswith("sentence-transformers")
    res = {"check": "embeddings", "backend": e.backend, "model": emb.MODEL_NAME, "offline_flag": os.getenv("HF_HUB_OFFLINE"),
           "paraphrase_pairs": rows, "semantic_separation_ok": sem_ok}
    if not is_st:
        res.update(result="FAIL", error="TF-IDF fallback is active: " + (e.fallback_reason or "unknown reason"))
    else:
        res["result"] = "PASS" if sem_ok else "FAIL"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["persistence", "embeddings"])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    res = check_persistence(a.live) if a.check == "persistence" else check_embeddings(a.download)
    _write(a.check + ("_live" if a.live else ""), res)
    return 0 if res["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
