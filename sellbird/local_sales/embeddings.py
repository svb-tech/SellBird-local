"""Local embeddings.

Primary: Sentence Transformers (all-MiniLM-L6-v2, Apache-2.0), loaded with
local_files_only so it can never reach the network at run time. If the package
or cached model is missing, a pure-Python TF-IDF cosine backend is used and
the backend name is reported in every result (so it is never presented as a
neural embedding).
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter

MODEL_NAME = os.getenv("SELLBIRD_EMBED_MODEL") or os.getenv("HARVEY_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or of to in for on with is are was be by at as it its this that from".split())


class Embedder:
    def __init__(self):
        self.backend = "tfidf-fallback"
        self.fallback_reason = ""
        self._model = None
        if (os.getenv("SELLBIRD_DISABLE_ST") or os.getenv("HARVEY_DISABLE_ST")) == "1":
            self.fallback_reason = "disabled by SELLBIRD_DISABLE_ST=1"
        else:
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(MODEL_NAME, local_files_only=True)
                self.backend = f"sentence-transformers:{MODEL_NAME}"
            except ImportError:
                self.fallback_reason = "sentence-transformers is not installed (pip install -r requirements-local.txt)"
            except Exception as e:  # model not cached, corrupt cache, ...
                self.fallback_reason = (f"model '{MODEL_NAME}' is not cached locally ({type(e).__name__}); run once "
                                        "with internet: python scripts/verify_local.py embeddings --download")

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity in [0, 1]."""
        if not a.strip() or not b.strip():
            return 0.0
        if self._model is not None:
            va, vb = self._model.encode([a, b], normalize_embeddings=True)
            return max(0.0, float(sum(x * y for x, y in zip(va, vb))))
        return _tfidf_cosine(a, b)


def _tokens(t: str) -> list[str]:
    return [w for w in _TOKEN.findall(t.lower()) if w not in _STOP and len(w) > 2]


def _tfidf_cosine(a: str, b: str) -> float:
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    vocab = set(ca) | set(cb)
    if not vocab:
        return 0.0
    idf = {w: math.log(3 / (1 + (w in ca) + (w in cb))) + 1 for w in vocab}
    va = {w: ca[w] * idf[w] for w in ca}
    vb = {w: cb[w] * idf[w] for w in cb}
    dot = sum(va[w] * vb.get(w, 0) for w in va)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0
