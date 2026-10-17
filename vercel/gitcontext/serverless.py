"""Query-side of GitContext for serverless platforms (Vercel).

Stateless: everything it needs lives in Qdrant (vectors, BM25-style sparse vectors, chunks, index metadata),
written there by `gitcontext index|sync` running elsewhere (laptop / GitHub Actions). No disk, no git, no
indexing here. Freshness = the commit recorded in Qdrant vs the repo's current HEAD from the GitHub API.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, HTTPException, Request

from . import __version__
from .config import Settings
from .index.embeddings import LazyEmbedder, make_embedder
from .index.qdrant_unified import QdrantUnified
from .llm import make_llm
from .rag import generate
from .retrieval.hybrid import Retriever
from .retrieval.rerank import make_reranker
from .schemas import QueryRequest

STATUS_TTL, HEAD_TTL = 30.0, 60.0


class OnlineEngine:
    def __init__(self, s: Settings, backend=None, embedder=None, llm="auto", head_provider=None, http_transport=None):
        self.s = s
        self.backend = backend or QdrantUnified(s.collection, url=s.qdrant_url, api_key=s.qdrant_api_key,
                                                in_memory=s.qdrant_in_memory)
        self.embedder = embedder or LazyEmbedder(lambda: make_embedder(s), f"{s.embedding_provider}:{s.embedding_model}")
        self.llm = make_llm(s) if llm == "auto" else llm
        reranker = make_reranker(s) if s.reranker == "lexical" else None  # no cross-encoder in the function
        self.retriever = Retriever(s, self.embedder, self.backend, self.backend, reranker)
        self._head_provider = head_provider or self._github_head
        self._http = httpx.Client(timeout=8, transport=http_transport)
        self._status: tuple[float, dict] | None = None
        self._head: tuple[float, str | None] | None = None
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Load the embedding model in the background so the first question doesn't pay for it."""
        if hasattr(self.embedder, "warm"):
            threading.Thread(target=self.embedder.warm, daemon=True, name="gitcontext-warm").start()

    # ---- freshness -------------------------------------------------------------------------------
    def _github_head(self) -> str | None:
        slug = self.s.github_slug
        if not slug:
            return None
        branch = self.backend.get_meta("branch") or "HEAD"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "gitcontext"}
        if self.s.github_token:
            headers["Authorization"] = f"Bearer {self.s.github_token}"
        try:
            r = self._http.get(f"https://api.github.com/repos/{slug}/commits/{branch}", headers=headers)
            return r.json()["sha"] if r.status_code == 200 else None
        except Exception:
            return None

    def head(self) -> str | None:
        now = time.time()
        if self._head and now - self._head[0] < HEAD_TTL:
            return self._head[1]
        h = self._head_provider()
        self._head = (now, h)
        return h

    def status(self) -> dict:
        now = time.time()
        if self._status and now - self._status[0] < STATUS_TTL:
            st = dict(self._status[1])
        else:
            b = self.backend
            indexed = b.get_meta("indexed_commit")
            st = {"repo": self.s.repo, "branch": b.get_meta("branch"), "indexed_commit": indexed,
                  "indexed_at": b.get_meta("indexed_at"), "embedder": b.get_meta("embedder"),
                  "indexed": bool(indexed), "chunks": b.counts_by_type() if indexed else {},
                  "vectors": b.count() if indexed else 0}
            self._status = (now, st)
        head = self.head() if st["indexed"] else None
        st.update(head=head or "", stale=bool(head and head != st["indexed_commit"]),
                  stale_files={"added": [], "modified": [], "deleted": [], "total": None},
                  job={"running": False, "last": None}, can_sync=False,
                  models={"embedder": st["embedder"] or self.embedder.name, "reranker": None,
                          "llm": getattr(self.llm, "name", None)})
        return st

    # ---- query --------------------------------------------------------------------------------------
    def _rate_limit(self, ip: str) -> None:
        limit = self.s.rate_limit_per_min
        if limit <= 0:
            return
        now = time.time()
        with self._lock:
            q = self._hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                raise HTTPException(429, f"rate limit: {limit} questions per minute")
            q.append(now)

    def search(self, req: QueryRequest):
        return self.retriever.search(req.question, mode=req.mode, rerank=False, top_k=req.top_k,
                                     source_types=req.source_types, path_prefix=req.path_prefix)

    def ask(self, req: QueryRequest, ip: str = "-") -> dict:
        self._rate_limit(ip)
        st = self.status()
        if not st["indexed"]:
            raise HTTPException(409, "nothing indexed yet: run the indexing workflow (see README) and retry")
        t0 = time.perf_counter()
        res = self.search(req)
        ans = generate(req.question, res.hits, self.llm, self.s.context_char_budget)
        ans.timings_ms.update(res.timings_ms)
        ans.timings_ms["total_ms"] = (time.perf_counter() - t0) * 1000
        ans.index_status = {"head": st["head"], "indexed_commit": st["indexed_commit"], "behind_head": st["stale"],
                            "indexed_at": st["indexed_at"], "stale_cited_files": []}
        return ans.to_dict()


def create_serverless_app(engine: OnlineEngine | None = None, settings: Settings | None = None) -> FastAPI:
    if os.environ.get("VERCEL"):  # read-only filesystem except /tmp
        os.environ.setdefault("HF_HOME", "/tmp/hf")
    s = settings or Settings()
    if os.environ.get("VERCEL") and not s.model_cache_dir:
        s.model_cache_dir = "/tmp/fastembed"
    state: dict = {"engine": engine}
    app = FastAPI(title="GitContext (serverless)", version=__version__)

    def eng() -> OnlineEngine:
        if state["engine"] is None:
            if not (s.qdrant_url or s.qdrant_in_memory):
                raise HTTPException(500, "GITCONTEXT_QDRANT_URL is not configured on the Vercel project")
            state["engine"] = OnlineEngine(s)
            state["engine"].warm()
        return state["engine"]

    def ip_of(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (request.client.host if request.client else "-")

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "mode": "serverless"}

    @app.get("/api/status")
    def status():
        try:
            return eng().status()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"cannot read the index from Qdrant: {type(e).__name__}: {e}")

    @app.post("/api/query")
    def query(req: QueryRequest, request: Request):
        try:
            return eng().ask(req, ip_of(request))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"query failed: {type(e).__name__}: {e}")

    @app.post("/api/search")
    def search(req: QueryRequest):
        res = eng().search(req)
        return {"mode": res.mode, "reranked": False, "timings_ms": res.timings_ms, "hits": [h.to_dict() for h in res.hits]}

    @app.post("/api/sync")
    @app.post("/api/index")
    def indexing_is_offline():
        raise HTTPException(403, "indexing runs in GitHub Actions for this deployment (.github/workflows/index.yml)")

    return app
