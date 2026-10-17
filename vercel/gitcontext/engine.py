"""Facade tying the components together. One Engine per repository."""
from __future__ import annotations

import hashlib
import hmac
import threading
import time

from .config import Settings
from .index.embeddings import make_embedder
from .index.indexer import Indexer
from .index.lexical import LexicalStore
from .index.qdrant_unified import QdrantUnified
from .index.vectorstore import make_store
from .ingest.repo import GitRepo
from .llm import make_llm
from .rag import Answer, generate
from .retrieval.hybrid import Retriever, SearchResult
from .retrieval.rerank import make_reranker


class Engine:
    def __init__(self, settings: Settings, *, embedder=None, store=None, lex=None, reranker="auto", llm="auto"):
        if not settings.repo:
            raise ValueError("No repository configured. Set GITCONTEXT_REPO or pass --repo.")
        self.s = settings
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        self.repo = GitRepo.open_or_clone(settings.repo, settings.work_dir, settings.github_token)
        self.embedder = embedder or make_embedder(settings)
        if settings.backend == "qdrant":
            # one object plays both roles: vectors + BM25-style sparse + chunks + manifest all live in Qdrant
            self.store = self.lex = store or QdrantUnified(
                settings.collection, url=settings.qdrant_url, api_key=settings.qdrant_api_key,
                path=str(settings.qdrant_path), in_memory=settings.qdrant_in_memory)
        else:
            self.store = store or make_store(settings)
            self.lex = lex or LexicalStore(settings.sqlite_path)
        self.reranker = make_reranker(settings) if reranker == "auto" else reranker
        self.llm = make_llm(settings) if llm == "auto" else llm
        self.indexer = Indexer(settings, self.repo, self.embedder, self.store, self.lex)
        self.retriever = Retriever(settings, self.embedder, self.store, self.lex, self.reranker)
        self.write_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_sync_report: dict | None = None

    # ---- indexing / freshness ---------------------------------------------------------------
    def index(self, **kw):
        with self.write_lock:
            return self.indexer.index_all(**kw)

    def sync(self, **kw):
        with self.write_lock:
            r = self.indexer.sync(**kw)
        self._last_sync_report = r.to_dict()
        return r

    def status(self) -> dict:
        st = self.indexer.status()
        st["last_sync"] = self._last_sync_report
        return st

    def start_auto_sync(self, interval: int) -> threading.Thread:
        def loop():
            while not self._stop.wait(interval):
                try:
                    self.sync()
                except Exception as e:  # keep the watcher alive
                    self._last_sync_report = {"error": str(e)}
        t = threading.Thread(target=loop, daemon=True, name="gitcontext-autosync")
        t.start()
        return t

    def stop(self):
        self._stop.set()

    def verify_webhook(self, body: bytes, signature: str | None) -> bool:
        if not self.s.webhook_secret:
            return False
        mac = "sha256=" + hmac.new(self.s.webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        return bool(signature) and hmac.compare_digest(mac, signature)

    # ---- query ------------------------------------------------------------------------------------
    def search(self, query: str, **kw) -> SearchResult:
        return self.retriever.search(query, **kw)

    def ask(self, question: str, *, top_k: int | None = None, mode: str = "hybrid", rerank: bool = True,
            source_types: list[str] | None = None, path_prefix: str | None = None) -> Answer:
        t0 = time.perf_counter()
        if self.s.sync_on_query and self.indexer.stale_files() != ([], [], []):
            self.sync(github=False)
        res = self.search(question, mode=mode, rerank=rerank, top_k=top_k, source_types=source_types,
                          path_prefix=path_prefix)
        ans = generate(question, res.hits, self.llm, self.s.context_char_budget)
        ans.timings_ms.update(res.timings_ms)
        ans.timings_ms["total_ms"] = (time.perf_counter() - t0) * 1000
        ans.index_status = self._freshness(ans)
        if ans.index_status.get("stale_cited_files"):
            ans.warnings.append("Some cited files changed since they were indexed: "
                                + ", ".join(ans.index_status["stale_cited_files"][:5]))
        return ans

    def _freshness(self, ans: Answer) -> dict:
        """Cheap staleness check at answer time: is HEAD ahead of the index, and are cited files outdated?"""
        head, indexed = self.repo.head(), self.lex.get_meta("indexed_commit")
        stale_cited: list[str] = []
        try:
            tree = self.repo.ls_tree("HEAD")
            for e in ans.evidence:
                p, b = e.get("path"), e.get("blob_sha")
                if p and b and (p not in tree or tree[p][0] != b) and p not in stale_cited:
                    stale_cited.append(p)
        except Exception:
            pass
        return {"head": head, "indexed_commit": indexed, "behind_head": head != indexed,
                "indexed_at": self.lex.get_meta("indexed_at"), "stale_cited_files": stale_cited}
