"""All-in-Qdrant backend: one object that is both the VectorStore and the lexical/chunk store.

Why: serverless functions have no persistent disk, so the SQLite FTS5 index cannot live next to them.
Here every chunk is one Qdrant point holding

  * a dense vector (semantic search),
  * a sparse BM25-style vector with server-side IDF (keyword search; see index/sparse.py),
  * the full chunk as payload (so retrieval needs no second store),
  * filterable fields: source_type, path prefixes, symbol leaf names, is_file.

The four rankers used by the hybrid retriever map to Qdrant like this:
  dense  -> query_points(using="dense")
  bm25   -> query_points(using="sparse")
  phrase -> sparse candidates (top 100) re-checked client-side for the exact/NEAR phrase
  symbol -> payload filter on `symbol_leaf`
The file manifest used for staleness is derived from point payloads (path + blob_sha), and index metadata
(indexed commit, embedder, ...) lives in a tiny companion collection, so any client can read freshness.
"""
from __future__ import annotations

import re
import uuid
from typing import Iterator

from ..models import NS, SOURCE_TYPES, Chunk
from .lexical import QueryProfile, query_tokens  # noqa: F401  (query_tokens re-exported for callers)
from .sparse import encode_chunk, encode_phrase_words, encode_query

DENSE, SPARSE = "dense", "sparse"
PHRASE_POOL = 100


def _leaf(symbol: str) -> list[str]:
    return [symbol.split(".")[-1].lower()] if symbol else []


class QdrantUnified:
    def __init__(self, collection: str, url: str | None = None, api_key: str | None = None,
                 path: str | None = None, in_memory: bool = False, client=None):
        from qdrant_client import QdrantClient

        self.collection, self.meta_collection = collection, f"{collection}_meta"
        self.remote = bool(url) and client is None
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        elif in_memory or not path:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=str(path))

    # ---- collection lifecycle -----------------------------------------------------------------
    def _exists(self, name: str) -> bool:
        return name in {c.name for c in self.client.get_collections().collections}

    def ensure(self, dim: int) -> None:
        from qdrant_client import models as qm

        if self._exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors[DENSE].size
            if existing != dim:
                raise RuntimeError(f"Collection '{self.collection}' has dim {existing} but the embedder produces "
                                   f"{dim}. Re-run `gitcontext index --rebuild` after changing the embedding model.")
        else:
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)})
            if self.remote:  # payload indexes only exist on a Qdrant server
                for f, schema in (("source_type", "keyword"), ("path_prefixes", "keyword"),
                                  ("symbol_leaf", "keyword"), ("symbol_full", "keyword"), ("is_file", "bool")):
                    try:
                        self.client.create_payload_index(self.collection, f, getattr(qm.PayloadSchemaType, schema.upper()))
                    except Exception:
                        pass
        if not self._exists(self.meta_collection):
            self.client.create_collection(self.meta_collection,
                                          vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))

    def reset(self) -> None:
        for name in (self.collection, self.meta_collection):
            if self._exists(name):
                self.client.delete_collection(name)

    # ---- writes ---------------------------------------------------------------------------------------
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict],
               chunks: list[Chunk] | None = None) -> None:
        from qdrant_client import models as qm

        if chunks is None:
            raise ValueError("QdrantUnified.upsert needs the chunks (it stores text + sparse vectors)")
        for i in range(0, len(ids), 64):
            pts = []
            for j in range(i, min(i + 64, len(ids))):
                c = chunks[j]
                idx, val = encode_chunk(c)
                payload = dict(payloads[j])
                payload.update(chunk=c.to_dict(), symbol_leaf=_leaf(c.symbol) if c.source_type == "code" else [],
                               symbol_full=[c.symbol.lower()] if c.source_type == "code" and c.symbol else [],
                               is_file=bool(c.path and c.blob_sha), blob_sha=c.blob_sha)
                pts.append(qm.PointStruct(id=ids[j], vector={DENSE: vectors[j], SPARSE: qm.SparseVector(indices=idx, values=val)},
                                          payload=payload))
            self.client.upsert(self.collection, points=pts)

    def delete(self, ids: list[str]) -> None:
        from qdrant_client import models as qm

        for i in range(0, len(ids), 500):
            self.client.delete(self.collection, points_selector=qm.PointIdsList(points=ids[i : i + 500]))

    def delete_source(self, source_type: str) -> list[str]:
        from qdrant_client import models as qm

        if self._exists(self.collection):
            self.client.delete(self.collection, points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="source_type", match=qm.MatchValue(value=source_type))])))
        return []

    # --- the LexicalStore write API is a no-op: everything was written by upsert() above -----------
    def set_file(self, *a, **k) -> None: ...
    def drop_file(self, *a, **k) -> None: ...

    # ---- meta ------------------------------------------------------------------------------------------
    def get_meta(self, k: str, default: str = "") -> str:
        try:
            r = self.client.retrieve(self.meta_collection, ids=[str(uuid.uuid5(NS, f"meta:{k}"))], with_payload=True)
        except Exception:
            return default
        return r[0].payload["v"] if r else default

    def set_meta(self, k: str, v: str) -> None:
        from qdrant_client import models as qm

        self.client.upsert(self.meta_collection, points=[qm.PointStruct(
            id=str(uuid.uuid5(NS, f"meta:{k}")), vector=[1.0], payload={"k": k, "v": v})])

    # ---- reads ----------------------------------------------------------------------------------------
    def _filter(self, source_types=None, path_prefix=None, extra=None):
        from qdrant_client import models as qm

        must = list(extra or [])
        if source_types:
            must.append(qm.FieldCondition(key="source_type", match=qm.MatchAny(any=list(source_types))))
        if path_prefix:
            must.append(qm.FieldCondition(key="path_prefixes", match=qm.MatchValue(value=path_prefix.rstrip("/"))))
        return qm.Filter(must=must) if must else None

    def search(self, vector, limit, source_types=None, path_prefix=None) -> list[tuple[str, float]]:
        res = self.client.query_points(self.collection, query=vector, using=DENSE, limit=limit,
                                       query_filter=self._filter(source_types, path_prefix), with_payload=False)
        return [(str(p.id), float(p.score)) for p in res.points]

    def _sparse(self, idx, val, limit, source_types, path_prefix, payload=False):
        from qdrant_client import models as qm

        if not idx:
            return []
        return self.client.query_points(
            self.collection, query=qm.SparseVector(indices=idx, values=val), using=SPARSE, limit=limit,
            query_filter=self._filter(source_types, path_prefix), with_payload=payload).points

    def bm25(self, profile: QueryProfile, k: int, source_types=None, path_prefix=None):
        idx, val = encode_query(profile)
        return [(str(p.id), float(p.score)) for p in self._sparse(idx, val, k, source_types, path_prefix)]

    def phrase(self, profile: QueryProfile, k: int, source_types=None, path_prefix=None):
        """Exact/NEAR phrase match (error messages), verified client-side on sparse candidates."""
        out: list[tuple[str, float]] = []
        for ph in profile.phrases:
            words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", ph)][:12]
            if len(words) < 2:
                continue
            idx, val = encode_phrase_words(words)
            for p in self._sparse(idx, val, PHRASE_POOL, source_types, path_prefix, payload=True):
                text = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", p.payload["chunk"]["text"])]
                if _has_phrase(text, words) and str(p.id) not in {i for i, _ in out}:
                    out.append((str(p.id), float(p.score)))
        return out[:k]

    def symbol_exact(self, profile: QueryProfile, k: int, source_types=None, path_prefix=None):
        """Chunks whose symbol equals a queried identifier. A qualified name (`Option.get_help_record`)
        ranks its exact match before same-named methods of other classes."""
        from qdrant_client import models as qm

        def run(field: str, names: list[str], limit: int):
            flt = self._filter(source_types, path_prefix,
                               extra=[qm.FieldCondition(key=field, match=qm.MatchAny(any=names))])
            pts, _ = self.client.scroll(self.collection, scroll_filter=flt, limit=limit, with_payload=False)
            return [str(p.id) for p in pts]

        qualified = list({i.lower() for i in profile.identifiers if "." in i})
        leaves = list({i.split(".")[-1].lower() for i in profile.identifiers})
        if not leaves:
            return []
        ids = run("symbol_full", qualified, k) if qualified else []
        ids += [i for i in run("symbol_leaf", leaves, k) if i not in ids]
        return [(i, 1.0) for i in ids[:k]]

    def get_many(self, ids: list[str]) -> dict[str, Chunk]:
        out: dict[str, Chunk] = {}
        for i in range(0, len(ids), 100):
            for p in self.client.retrieve(self.collection, ids=ids[i : i + 100], with_payload=["chunk"]):
                out[str(p.id)] = Chunk.from_dict(p.payload["chunk"])
        return out

    # ---- inventory (manifest / counts / iteration) --------------------------------------------------
    def _scroll(self, flt=None, payload=True) -> Iterator:
        offset = None
        while True:
            pts, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=500, offset=offset,
                                             with_payload=payload, with_vectors=False)
            yield from pts
            if offset is None:
                return

    def manifest(self) -> dict[str, tuple[str, list[str]]]:
        from qdrant_client import models as qm

        if not self._exists(self.collection):
            return {}
        flt = qm.Filter(must=[qm.FieldCondition(key="is_file", match=qm.MatchValue(value=True))])
        files: dict[str, tuple[str, list[str]]] = {}
        for p in self._scroll(flt, payload=["path", "blob_sha"]):
            path, sha = p.payload["path"], p.payload["blob_sha"]
            files.setdefault(path, (sha, []))[1].append(str(p.id))
        return files

    def count(self, source_type: str | None = None) -> int:
        from qdrant_client import models as qm

        flt = (qm.Filter(must=[qm.FieldCondition(key="source_type", match=qm.MatchValue(value=source_type))])
               if source_type else None)
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def counts_by_type(self) -> dict[str, int]:
        if not self._exists(self.collection):
            return {}
        return {st: n for st in SOURCE_TYPES if (n := self.count(st))}

    def iter_chunks(self, source_type: str | None = None) -> Iterator[Chunk]:
        from qdrant_client import models as qm

        flt = (qm.Filter(must=[qm.FieldCondition(key="source_type", match=qm.MatchValue(value=source_type))])
               if source_type else None)
        for p in self._scroll(flt, payload=["chunk"]):
            yield Chunk.from_dict(p.payload["chunk"])


def _has_phrase(text: list[str], words: list[str]) -> bool:
    n = len(words)
    if n > len(text):
        return False
    for i in range(len(text) - n + 1):  # contiguous match
        if text[i : i + n] == words:
            return True
    pos: dict[str, list[int]] = {}
    for i, t in enumerate(text):
        if t in words:
            pos.setdefault(t, []).append(i)
    if len(pos) < len(set(words)):
        return False
    first = min(pos.values(), key=len)  # NEAR: all words within a window of len + 8 tokens
    return any(all(any(abs(q - s) <= n + 8 for q in pos[w]) for w in pos) for s in first)
