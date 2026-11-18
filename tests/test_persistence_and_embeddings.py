import importlib.util
import sys
import types
from pathlib import Path

import pytest

from sellbird.local_sales import embeddings as emb

ROOT = Path(__file__).resolve().parent.parent


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_local", ROOT / "scripts" / "verify_local.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sqlite_persistence_survives_restart_in_a_new_process():
    res = _load_verify().check_persistence(live=False)
    assert res["result"] == "PASS", res
    assert res["queued_task_survived_restart"] and res["integrity_check"] == "ok"
    assert res["rows_after_restart"] == res["rows_written"]


def _fake_st(monkeypatch, behaviour):
    calls = {}

    class FakeModel:
        def __init__(self, name, **kw):
            calls.update(name=name, kw=kw)
            if behaviour == "uncached":
                raise OSError("not in local cache")

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0], [1.0, 0.0]]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    monkeypatch.delenv("SELLBIRD_DISABLE_ST", raising=False)
    monkeypatch.delenv("HARVEY_DISABLE_ST", raising=False)
    return calls


def test_embedder_uses_sentence_transformers_offline_only(monkeypatch):
    calls = _fake_st(monkeypatch, "ok")
    e = emb.Embedder()
    assert e.backend == f"sentence-transformers:{emb.MODEL_NAME}" and e.fallback_reason == ""
    assert calls["kw"] == {"local_files_only": True}  # can never download at run time
    assert e.similarity("a b", "c d") == pytest.approx(1.0)


def test_embedder_reports_why_it_fell_back(monkeypatch):
    _fake_st(monkeypatch, "uncached")
    e = emb.Embedder()
    assert e.backend == "tfidf-fallback" and "not cached" in e.fallback_reason and "--download" in e.fallback_reason
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import fails
    assert "not installed" in emb.Embedder().fallback_reason
    monkeypatch.setenv("SELLBIRD_DISABLE_ST", "1")
    assert "disabled" in emb.Embedder().fallback_reason
