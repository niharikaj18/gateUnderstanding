"""FastAPI backend.

POST /query            ask a question (hybrid retrieval -> rerank -> LLM with citations)
POST /search           retrieval only, with per-ranker rank explanations
GET  /status           index freshness (HEAD vs indexed commit, stale files)
POST /index, /sync     (re)index everything / only what changed
POST /webhook/github   push-event webhook -> background sync (HMAC verified)
"""
from __future__ import annotations

import hmac
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings
from .engine import Engine
from .schemas import IndexRequest, QueryRequest


def create_app(engine: Engine | None = None, settings: Settings | None = None) -> FastAPI:
    state: dict = {"engine": engine, "job": {"running": False, "last": None}}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            state["engine"] = Engine(settings or Settings())
        eng: Engine = state["engine"]
        if eng.s.auto_index_on_start and not eng.status()["indexed"]:
            threading.Thread(target=run_job, args=(eng.index,), kwargs={"github": True}, daemon=True,
                             name="gitcontext-initial-index").start()
        if eng.s.auto_sync_seconds > 0:
            eng.start_auto_sync(eng.s.auto_sync_seconds)
        yield
        eng.stop()

    OPEN_PATHS = {"/health", "/webhook/github", "/docs", "/openapi.json", "/redoc"}

    def require_key(request: Request):
        """Shared-secret auth. The webhook has its own HMAC; /health stays open for platform probes."""
        key = eng().s.api_key if state["engine"] else None
        if not key or request.url.path in OPEN_PATHS or request.method == "OPTIONS":
            return
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), key):
            raise HTTPException(401, "missing or invalid x-api-key")

    app = FastAPI(title="GitContext", version=__version__, lifespan=lifespan, dependencies=[Depends(require_key)])
    cfg = engine.s if engine else (settings or Settings())
    origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["content-type", "x-api-key"])

    def eng() -> Engine:
        return state["engine"]

    def run_job(fn, **kw):
        job = state["job"]
        if job["running"]:
            return
        job["running"] = True
        try:
            job["last"] = fn(**kw).to_dict()
        except Exception as e:
            job["last"] = {"error": str(e)}
        finally:
            job["running"] = False

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/status")
    def status():
        st = eng().status()
        st["job"] = state["job"]
        st["models"] = {"embedder": eng().embedder.name, "reranker": getattr(eng().reranker, "name", None),
                        "llm": getattr(eng().llm, "name", None)}
        return st

    @app.post("/index", status_code=202)
    def index(req: IndexRequest, bg: BackgroundTasks):
        if state["job"]["running"]:
            raise HTTPException(409, "an indexing job is already running")
        bg.add_task(run_job, eng().index, **req.model_dump())
        return {"started": True}

    @app.post("/sync")
    def sync(wait: bool = True, bg: BackgroundTasks = None):
        if state["job"]["running"]:
            raise HTTPException(409, "an indexing job is already running")
        if not wait:
            bg.add_task(run_job, eng().sync)
            return {"started": True}
        run_job(eng().sync)
        return state["job"]["last"]

    @app.post("/search")
    def search(req: QueryRequest):
        res = eng().search(req.question, mode=req.mode, rerank=req.rerank, top_k=req.top_k,
                           source_types=req.source_types, path_prefix=req.path_prefix)
        return {"mode": res.mode, "reranked": res.reranked, "timings_ms": res.timings_ms,
                "hits": [h.to_dict() for h in res.hits]}

    @app.post("/query")
    def query(req: QueryRequest):
        if not eng().status()["indexed"]:
            raise HTTPException(409, "repository not indexed yet: POST /index first")
        return eng().ask(req.question, top_k=req.top_k, mode=req.mode, rerank=req.rerank,
                         source_types=req.source_types, path_prefix=req.path_prefix).to_dict()

    @app.post("/webhook/github")
    async def webhook(request: Request, bg: BackgroundTasks):
        body = await request.body()
        if not eng().verify_webhook(body, request.headers.get("x-hub-signature-256")):
            raise HTTPException(401, "bad signature")
        event = request.headers.get("x-github-event", "")
        if event == "ping":
            return {"pong": True}
        if event in ("push", "pull_request", "issues", "issue_comment", "pull_request_review_comment"):
            bg.add_task(run_job, eng().sync)
            return {"sync_scheduled": True, "event": event}
        return {"ignored": event}

    return app


def app_factory() -> FastAPI:  # uvicorn --factory gitcontext.api:app_factory
    return create_app()
