"""Rerankers. A cross-encoder reads (query, passage) jointly, which fixes the ordering mistakes
bi-encoder similarity and BM25 make on near-miss candidates."""
from __future__ import annotations

import re
from typing import Protocol

from ..config import Settings
from ..index.lexical import expand_identifiers, profile_query
from ..models import Chunk


class Reranker(Protocol):
    name: str

    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str, max_chars: int = 1200):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = f"cross-encoder:{model}"
        self.max_chars = max_chars
        self._m = TextCrossEncoder(model)

    def score(self, query, chunks):
        docs = [c.embed_text()[: self.max_chars] for c in chunks]
        return [float(x) for x in self._m.rerank(query, docs)]


class LexicalReranker:
    """Dependency-free fallback: weighted term overlap with symbol/path boosts."""

    name = "lexical"

    def score(self, query, chunks):
        prof = profile_query(query)
        terms = set(prof.terms) | {t.lower() for i in prof.identifiers for t in re.findall(r"[A-Za-z0-9]+", i)}
        out = []
        for c in chunks:
            body = set(re.findall(r"[a-z0-9_]+", (c.text + " " + expand_identifiers(c.text)).lower()))
            sym = set(re.findall(r"[a-z0-9]+", (c.symbol + " " + c.path + " " + expand_identifiers(c.symbol)).lower()))
            s = sum(1.0 for t in terms if t in body) / (len(terms) or 1) + 1.5 * sum(1.0 for t in terms if t in sym) / (len(terms) or 1)
            if any(ph.lower() in c.text.lower() for ph in prof.phrases):
                s += 2.0
            out.append(s)
        return out


def make_reranker(s: Settings) -> Reranker | None:
    if s.reranker == "none":
        return None
    if s.reranker == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(s.reranker_model, s.rerank_max_chars)
