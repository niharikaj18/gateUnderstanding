"""Vector store abstraction + Qdrant implementation (embedded local mode, in-memory, or server).

Swapping in Pinecone/Weaviate means implementing the five methods of `VectorStore`;
nothing else in the codebase touches the vector DB directly.
"""
from __future__ import annotations

from typing import Protocol

from ..config import Settings


class VectorStore(Protocol):
    def ensure(self, dim: int) -> None: ...
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search(self, vector: list[float], limit: int, source_types=None, path_prefix=None) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


class QdrantStore:
    def __init__(self, collection: str, url: str | None = None, api_key: str | None = None,
                 path: str | None = None, in_memory: bool = False):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.remote = bool(url)
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif in_memory or not path:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=str(path))

    def ensure(self, dim: int) -> None:
        from qdrant_client import models as qm

        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors.size
            if existing != dim:
                raise RuntimeError(f"Collection '{self.collection}' has dim {existing} but embedder produces {dim}. "
                                   "Re-run `gitcontext index --rebuild` after changing the embedding model.")
            return
        self.client.create_collection(self.collection,
                                      vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
        if not self.remote:  # payload indexes only exist on a Qdrant server
            return
        for f in ("source_type", "path_prefixes", "language", "author"):
            try:
                self.client.create_payload_index(self.collection, f, qm.PayloadSchemaType.KEYWORD)
            except Exception:  # local mode ignores payload indexes
                pass

    def reset(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            self.client.delete_collection(self.collection)

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client import models as qm

        for i in range(0, len(ids), 128):
            self.client.upsert(self.collection, points=[
                qm.PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j])
                for j in range(i, min(i + 128, len(ids)))])

    def delete(self, ids) -> None:
        from qdrant_client import models as qm

        for i in range(0, len(ids), 500):
            self.client.delete(self.collection, points_selector=qm.PointIdsList(points=ids[i : i + 500]))

    def search(self, vector, limit, source_types=None, path_prefix=None):
        from qdrant_client import models as qm

        must = []
        if source_types:
            must.append(qm.FieldCondition(key="source_type", match=qm.MatchAny(any=list(source_types))))
        if path_prefix:
            must.append(qm.FieldCondition(key="path_prefixes", match=qm.MatchValue(value=path_prefix.rstrip("/"))))
        res = self.client.query_points(self.collection, query=vector, limit=limit,
                                       query_filter=qm.Filter(must=must) if must else None, with_payload=False)
        return [(str(p.id), float(p.score)) for p in res.points]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count


def make_store(s: Settings) -> QdrantStore:
    s.work_dir.mkdir(parents=True, exist_ok=True)
    return QdrantStore(s.collection, url=s.qdrant_url, api_key=s.qdrant_api_key,
                       path=str(s.qdrant_path), in_memory=s.qdrant_in_memory)
