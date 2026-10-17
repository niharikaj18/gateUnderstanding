"""Hybrid retrieval: dense vectors + BM25 + exact-phrase + exact-symbol, fused with Reciprocal Rank
Fusion, then (optionally) reranked by a cross-encoder and de-duplicated per file."""
from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..config import Settings
from ..index.embeddings import Embedder
from ..index.lexical import LexicalStore, profile_query
from ..index.vectorstore import VectorStore
from ..models import Chunk
from .rerank import Reranker

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    fused: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # ranker name -> 1-based rank (explainability)
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d.update(score=self.score, fused_score=self.fused, ranks=self.ranks, rerank_score=self.rerank_score)
        return d


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]
    mode: str
    reranked: bool


def rrf(rankings: dict[str, list[str]], weights: dict[str, float]) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, ids in rankings.items():
        w = weights.get(name, 1.0)
        for r, cid in enumerate(ids, 1):
            scores[cid] += w / (RRF_K + r)
            ranks[cid][name] = r
    return scores, ranks


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore, lex: LexicalStore,
                 reranker: Reranker | None):
        self.s, self.embedder, self.store, self.lex, self.reranker = settings, embedder, store, lex, reranker

    def search(self, query: str, *, mode: str = "hybrid", rerank: bool = True, top_k: int | None = None,
               source_types: list[str] | None = None, path_prefix: str | None = None) -> SearchResult:
        s, top_k = self.s, top_k or self.s.final_k
        t: dict[str, float] = {}
        T0 = time.perf_counter()
        rankings: dict[str, list[str]] = {}
        prof = profile_query(query)

        # Over a network (Qdrant Cloud) each ranker is a round trip, so run them concurrently.
        parallel = bool(getattr(self.store, "remote", False)) and self.lex is self.store
        pool = ThreadPoolExecutor(max_workers=3) if parallel and mode != "dense" else None

        def timed(name, fn):
            t0 = time.perf_counter()
            res = [i for i, _ in fn()]
            t[name] = (time.perf_counter() - t0) * 1000
            return res

        lex_jobs = {}
        if mode in ("bm25", "hybrid"):
            jobs = {"bm25": lambda: self.lex.bm25(prof, s.retrieve_k, source_types, path_prefix),
                    "phrase": lambda: self.lex.phrase(prof, s.retrieve_k, source_types, path_prefix),
                    "symbol": lambda: self.lex.symbol_exact(prof, 10, source_types, path_prefix)}
            if pool:
                lex_jobs = {n: pool.submit(timed, f"lex_{n}", fn) for n, fn in jobs.items()}
            else:
                t0 = time.perf_counter()
                for n, fn in jobs.items():
                    rankings[n] = [i for i, _ in fn()]
                t["lexical"] = (time.perf_counter() - t0) * 1000
        if mode in ("dense", "hybrid"):
            t0 = time.perf_counter()
            qv = self.embedder.embed_query(query)
            t["embed_query"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            rankings["dense"] = [i for i, _ in self.store.search(qv, s.retrieve_k, source_types, path_prefix)]
            t["dense"] = (time.perf_counter() - t0) * 1000
        for n, fut in lex_jobs.items():
            rankings[n] = fut.result()
        if pool:
            pool.shutdown()
            t["lexical"] = max(t.pop(f"lex_{n}") for n in lex_jobs)

        # A qualified name like `Option.get_help_record` is an unambiguous request: trust the symbol ranker.
        qualified = any("." in i for i in prof.identifiers)
        weights = {"dense": 1.0, "bm25": 1.0 + (0.5 if prof.exact_signal else 0.0),
                   "phrase": 2.0, "symbol": 4.0 if qualified else 1.5}
        t0 = time.perf_counter()
        fused, ranks = rrf({k: v for k, v in rankings.items() if v}, weights)
        order = sorted(fused, key=fused.get, reverse=True)
        pool_n = s.rerank_k if (rerank and self.reranker) else max(top_k * 3, top_k)
        pool_ids = order[:pool_n]
        chunks = self.lex.get_many(pool_ids)
        hits = [Hit(chunks[i], fused[i], fused[i], ranks[i]) for i in pool_ids if i in chunks]
        t["fuse"] = (time.perf_counter() - t0) * 1000

        reranked = False
        if rerank and self.reranker and len(hits) > 1:
            t0 = time.perf_counter()
            scores = self.reranker.score(query, [h.chunk for h in hits])
            for h, sc in zip(hits, scores):
                h.rerank_score, h.score = sc, sc
            a = s.rerank_blend
            if a < 1.0:
                # Blend: don't let a general-purpose reranker fully override exact-match evidence.
                # `hits` is currently in fused order; combine fused rank and reranker rank with RRF.
                rr_rank = {id(h): r for r, h in enumerate(sorted(hits, key=lambda h: h.rerank_score, reverse=True), 1)}
                for fr, h in enumerate(hits, 1):
                    h.score = (1 - a) / (RRF_K + fr) + a / (RRF_K + rr_rank[id(h)])
            hits.sort(key=lambda h: h.score, reverse=True)
            reranked = True
            t["rerank"] = (time.perf_counter() - t0) * 1000

        hits = self._diversify(hits, top_k)
        t["total"] = (time.perf_counter() - T0) * 1000
        return SearchResult(hits, t, mode, reranked)

    def _diversify(self, hits: list[Hit], k: int) -> list[Hit]:
        """Cap chunks per file so one big file can't crowd out commits/PRs/docs."""
        per_file: dict[str, int] = defaultdict(int)
        out: list[Hit] = []
        for h in hits:
            key = h.chunk.path or h.chunk.key
            if per_file[key] >= self.s.max_chunks_per_file:
                continue
            per_file[key] += 1
            out.append(h)
            if len(out) >= k:
                break
        return out
