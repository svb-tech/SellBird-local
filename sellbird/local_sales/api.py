"""FastAPI router + standalone app for the local sales agent."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sellbird.local_sales import netguard, workflows as wf
from sellbird.local_sales.embeddings import Embedder
from sellbird.local_sales.hardware import PROFILES, recommend_model
from sellbird.local_sales.llm import LocalLLM, read_inference_log
from sellbird.local_sales.store import Store

router = APIRouter(prefix="/api/local", tags=["local-sales"])
_ctx: wf.Ctx | None = None
UI_FILE = Path(__file__).resolve().parent.parent / "web" / "local.html"


def get_ctx() -> wf.Ctx:
    global _ctx
    if _ctx is None:
        rec = recommend_model()
        prof = PROFILES[rec["profile"]]
        model = os.getenv("SELLBIRD_LOCAL_MODEL") or os.getenv("HARVEY_LOCAL_MODEL") or "qwen3:1.7b"
        _ctx = wf.Ctx(Store(), LocalLLM(model=model, num_ctx=prof["num_ctx"], num_predict=prof["num_predict"]),
                      Embedder())
    return _ctx


class Payload(BaseModel):
    prospect_id: int | None = None
    product_id: int | None = None
    text: str | None = None


class PipelineReq(BaseModel):
    product_id: int
    max_prospects: int = 6


@router.get("/status")
def status():
    c = get_ctx()
    rec = recommend_model()
    log = read_inference_log(1000)
    return {
        "ollama": c.llm.health(), "model_in_use": c.llm.model, "recommendation": rec,
        "embedding_backend": c.embedder.backend, "embedding_fallback_reason": c.embedder.fallback_reason, "queue_size": len(c.store.queue("queued")),
        "inference_calls_logged": len(log),
        "all_logged_calls_local": all(r["endpoint"].startswith(("http://127.0.0.1", "http://localhost")) or
                                      os.getenv("LOCAL_AI_ALLOW_PRIVATE_HOST") == "1" for r in log),
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in log),
        "network_guard": netguard.report(),
    }


@router.post("/setup/load")
def setup_load():
    c = get_ctx()
    try:
        a = wf.ingest_company_files(c.store)
        b = wf.load_prospects(c.store)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"company": a, "prospects": b, "method": "rules (file ingestion, no AI)"}


@router.post("/business/profile")
def business_profile():
    return wf.run_workflow(get_ctx(), "business_profile", {})


@router.get("/business")
def business():
    return get_ctx().store.business() or {}


@router.get("/products")
def products():
    return get_ctx().store.products()


@router.post("/products/{pid}/icp")
def product_icp(pid: int):
    return wf.run_workflow(get_ctx(), "icp", {"product_id": pid})


@router.get("/prospects")
def prospects():
    return [{k: v for k, v in p.items() if k != "raw_notes"} for p in get_ctx().store.prospects()]


@router.get("/prospects/{pid}")
def prospect_detail(pid: int, product_id: int):
    s = get_ctx().store
    return {"prospect": s.prospect(pid),
            "signals": s.q("SELECT * FROM signals WHERE prospect_id=? AND product_id=?", (pid, product_id)),
            "problem": s.one("SELECT * FROM problems WHERE prospect_id=? AND product_id=?", (pid, product_id)),
            "score": s.one("SELECT * FROM scores WHERE prospect_id=? AND product_id=?", (pid, product_id)),
            "emails": s.q("SELECT * FROM emails WHERE prospect_id=? AND product_id=? ORDER BY id DESC",
                          (pid, product_id))}


@router.post("/run/{workflow}")
def run(workflow: str, body: Payload):
    if workflow not in wf.WORKFLOWS:
        raise HTTPException(404, f"Unknown workflow. Use one of {list(wf.WORKFLOWS)}")
    return wf.run_workflow(get_ctx(), workflow, body.model_dump(exclude_none=True))


@router.post("/pipeline")
def pipeline(req: PipelineReq):
    return wf.run_pipeline(get_ctx(), req.product_id, req.max_prospects)


@router.get("/ranking")
def ranking(product_id: int):
    return get_ctx().store.q(
        "SELECT p.id AS prospect_id, p.company, p.title, s.total, s.tier, s.breakdown_json FROM scores s "
        "JOIN prospects p ON p.id=s.prospect_id WHERE s.product_id=? ORDER BY s.total DESC", (product_id,))


@router.post("/replies/evaluate")
def replies_evaluate():
    return wf.evaluate_reply_classifier(get_ctx())


@router.get("/inference-log")
def inference_log(limit: int = 50):
    return list(reversed(read_inference_log(limit)))


@router.get("/queue")
def queue():
    return get_ctx().store.queue()


@router.post("/queue/process")
def queue_process():
    return wf.process_queue(get_ctx())


@router.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


def create_app() -> FastAPI:
    app = FastAPI(title="Sellbird Local Sales Agent")
    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    def root():
        return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))

    return app
