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
    def __init__(self, model: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        self._m = TextEmbedding(model, cache_dir=cache_dir) if cache_dir else TextEmbedding(model)
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


class VoyageEmbedder:
    """Voyage AI over plain HTTP (no SDK; light enough for serverless). `voyage-code-3` is code-specialised.
    Free accounts without a payment method are limited to 3 requests/min, so we batch and retry on 429."""

    URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, model: str, api_key: str | None, dim: int | None = None, transport=None):
        import httpx

        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required for the voyage embedding provider")
        self.name = f"voyage:{model}" + (f":{dim}" if dim else "")
        self._model, self._dim_req = model, dim
        self._http = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key}"}, transport=transport)
        self.dim = len(self.embed_query("probe"))

    def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        import time

        body = {"input": texts, "model": self._model, "input_type": input_type}
        if self._dim_req:
            body["output_dimension"] = self._dim_req
        for attempt in range(8):
            r = self._http.post(self.URL, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(float(r.headers.get("retry-after", 0)) or 2 ** attempt * 3, 65))
                continue
            r.raise_for_status()
            return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        raise RuntimeError("Voyage API kept rate-limiting; add a payment method (free tokens still apply) or slow down")

    def embed_documents(self, texts):
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            out += self._call([t[:6000] for t in texts[i : i + 64]], "document")
        return out

    def embed_query(self, text):
        return self._call([text[:6000]], "query")[0]


class LazyEmbedder:
    """Defers loading a heavy model until first use (serverless cold starts) and lets it be warmed in a thread."""

    def __init__(self, factory, name_hint: str):
        import threading

        self._factory, self._inner, self._lock, self.name_hint = factory, None, threading.Lock(), name_hint

    def _get(self):
        with self._lock:
            if self._inner is None:
                self._inner = self._factory()
            return self._inner

    @property
    def name(self):
        return self._inner.name if self._inner else self.name_hint

    @property
    def dim(self):
        return self._get().dim

    def embed_documents(self, texts):
        return self._get().embed_documents(texts)

    def embed_query(self, text):
        return self._get().embed_query(text)

    def warm(self):
        self._get()


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
    if s.embedding_provider == "voyage":
        model = s.embedding_model if s.embedding_model.startswith("voyage") else "voyage-code-3"
        return VoyageEmbedder(model, s.voyage_api_key, s.embedding_dim)
    return FastEmbedder(s.embedding_model, s.model_cache_dir)
