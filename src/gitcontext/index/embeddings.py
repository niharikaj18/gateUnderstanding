"""Embedding providers behind one interface.

* fastembed – local ONNX models (default BAAI/bge-small-en-v1.5), no API key, no torch
* openai    – text-embedding-3-small/large (or any OpenAI-compatible endpoint)
* hash      – deterministic feature-hashing embedder for offline tests/CI (NOT semantic)
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from ..config import Settings
from .lexical import expand_identifiers


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        self._m = TextEmbedding(model)
        self.dim = len(next(iter(self._m.embed(["probe"]))))

    def embed_documents(self, texts):
        return [v.tolist() for v in self._m.embed(texts, batch_size=32)]

    def embed_query(self, text):
        return next(iter(self._m.query_embed([text]))).tolist()


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str | None, base_url: str | None):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self._c = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.dim = len(self.embed_query("probe"))

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), 96):
            r = self._c.embeddings.create(model=self._model, input=[t[:8000] for t in texts[i : i + 96]])
            out += [d.embedding for d in r.data]
        return out

    def embed_query(self, text):
        return self._c.embeddings.create(model=self._model, input=[text[:8000]]).data[0].embedding


class HashEmbedder:
    """Signed feature hashing over identifier-expanded tokens. Lexical-ish, deterministic, instant."""

    def __init__(self, dim: int = 384):
        self.name, self.dim = f"hash:{dim}", dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9_]+", (text + " " + expand_identifiers(text)).lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_embedder(s: Settings) -> Embedder:
    if s.embedding_provider == "hash":
        return HashEmbedder()
    if s.embedding_provider == "openai":
        model = s.embedding_model if s.embedding_model.startswith("text-embedding") else "text-embedding-3-small"
        return OpenAIEmbedder(model, s.openai_api_key, s.openai_base_url)
    return FastEmbedder(s.embedding_model)
