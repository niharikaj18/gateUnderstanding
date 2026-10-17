import os

import httpx
import pytest
from conftest import git
from fastapi.testclient import TestClient

from gitcontext.config import Settings
from gitcontext.engine import Engine
from gitcontext.index.embeddings import LazyEmbedder
from gitcontext.serverless import OnlineEngine, create_serverless_app

SERVER = os.environ.get("GITCONTEXT_TEST_QDRANT_URL")


@pytest.fixture()
def indexed(repo, tmp_path):
    """An all-in-Qdrant index written by the indexer (as the GitHub Action would)."""
    s = Settings(repo=str(repo), data_dir=tmp_path / "d", embedding_provider="hash", reranker="none",
                 llm_provider="none", backend="qdrant", qdrant_in_memory=True, github_token=None)
    e = Engine(s, llm=None)
    e.index(github=False)
    return e


def online(indexed, **over):
    s = indexed.s.model_copy(update=over)
    return OnlineEngine(s, backend=indexed.store, embedder=indexed.embedder, llm=None,
                        head_provider=lambda: indexed.repo.head())


def client(eng):
    return TestClient(create_serverless_app(engine=eng, settings=eng.s))


def test_status_contract_and_freshness(indexed, repo):
    c = client(online(indexed))
    st = c.get("/api/status").json()
    assert st["indexed"] and not st["stale"] and st["can_sync"] is False
    assert st["stale_files"]["total"] is None and st["chunks"]["code"] > 0
    assert st["indexed_commit"] == indexed.repo.head()
    # a new commit lands: the serverless side notices (HEAD != indexed commit) without any git access
    (repo / "billing" / "payments.py").write_text((repo / "billing" / "payments.py").read_text() + "\n# change\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "change")
    eng = online(indexed)
    st = client(eng).get("/api/status").json()
    assert st["stale"] and st["head"] != st["indexed_commit"]
    a = client(eng).post("/api/query", json={"question": "where is refund_order defined?"}).json()
    assert a["index_status"]["behind_head"] is True and a["evidence"]


def test_query_search_and_filters(indexed):
    c = client(online(indexed))
    a = c.post("/api/query", json={"question": 'where does "payment gateway unreachable after 3 attempts" come from', "top_k": 3}).json()
    assert a["evidence"][0]["symbol"] == "charge_customer" and a["model"] == "extractive"
    r = c.post("/api/search", json={"question": "refund_order", "mode": "bm25", "top_k": 3}).json()
    assert r["hits"][0]["symbol"] == "refund_order" and r["reranked"] is False
    r = c.post("/api/search", json={"question": "idempotency", "source_types": ["incident"]}).json()
    assert r["hits"] and all(h["source_type"] == "incident" for h in r["hits"])


def test_indexing_endpoints_are_disabled(indexed):
    c = client(online(indexed))
    assert c.post("/api/sync").status_code == 403 and c.post("/api/index").status_code == 403


def test_rate_limit_per_ip(indexed):
    c = client(online(indexed, rate_limit_per_min=2))
    body = {"question": "refund_order"}
    assert [c.post("/api/query", json=body, headers={"x-forwarded-for": "1.2.3.4"}).status_code for _ in range(3)] == [200, 200, 429]
    assert c.post("/api/query", json=body, headers={"x-forwarded-for": "5.6.7.8"}).status_code == 200


def test_not_indexed_and_misconfigured(tmp_path):
    empty = Settings(repo="o/r", embedding_provider="hash", qdrant_in_memory=True, reranker="none", llm_provider="none", data_dir=tmp_path)
    from gitcontext.index.qdrant_unified import QdrantUnified
    be = QdrantUnified("c", in_memory=True); be.ensure(384)  # collection exists, no meta
    c = client(OnlineEngine(empty, backend=be, embedder=LazyEmbedder(lambda: None, "x"), llm=None, head_provider=lambda: None))
    assert c.get("/api/status").json()["indexed"] is False
    r = c.post("/api/query", json={"question": "anything"})
    assert r.status_code == 409 and "indexed" in r.json()["detail"]
    bare = TestClient(create_serverless_app(settings=Settings(repo="o/r")))
    r = bare.get("/api/status")
    assert r.status_code == 500 and "QDRANT_URL" in r.json()["detail"]


def test_github_head_lookup_and_failure(indexed):
    eng = OnlineEngine(indexed.s.model_copy(update={"repo": "o/r"}), backend=indexed.store, embedder=indexed.embedder, llm=None,
                       http_transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"sha": "f" * 40})
                                                          if req.url.path.endswith("/repos/o/r/commits/main") or "commits" in req.url.path
                                                          else httpx.Response(404)))
    assert eng.head() == "f" * 40
    down = OnlineEngine(indexed.s.model_copy(update={"repo": "o/r"}), backend=indexed.store, embedder=indexed.embedder, llm=None,
                        http_transport=httpx.MockTransport(lambda req: httpx.Response(403)))
    assert down.head() is None and down.status()["stale"] is False  # unknown HEAD is never reported as stale


def test_lazy_embedder_defers_loading():
    calls = []
    lazy = LazyEmbedder(lambda: calls.append(1) or type("E", (), {"name": "m", "dim": 3, "embed_query": lambda self, t: [0.0] * 3})(), "hint")
    assert lazy.name == "hint" and not calls
    assert lazy.embed_query("x") == [0.0] * 3 and calls == [1] and lazy.name == "m"


@pytest.mark.skipif(not SERVER, reason="needs GITCONTEXT_TEST_QDRANT_URL (a real Qdrant server)")
def test_stateless_reader_sees_what_the_indexer_wrote(repo, tmp_path):
    """Indexer (GitHub Action) and reader (Vercel) share nothing but the Qdrant server."""
    base = dict(repo=str(repo), embedding_provider="hash", reranker="none", llm_provider="none", backend="qdrant",
                qdrant_url=SERVER, collection_name="gc_e2e_" + tmp_path.name.lower().replace("-", "")[:20], github_token=None)
    writer = Engine(Settings(data_dir=tmp_path / "w", **base), llm=None)
    writer.index(github=False)
    reader = OnlineEngine(Settings(data_dir=tmp_path / "r", **base), head_provider=lambda: writer.repo.head())  # brand-new client
    c = client(reader)
    assert c.get("/api/status").json()["indexed"] is True
    a = c.post("/api/query", json={"question": "refund_order"}).json()
    assert a["evidence"][0]["symbol"] == "refund_order" and a["evidence"][0]["author"] == "Alice"
    # the Action re-runs `sync` after a commit; the reader sees the new code without any redeploy
    (repo / "billing" / "payments.py").write_text((repo / "billing" / "payments.py").read_text() + "\ndef brand_new_fn():\n    return 1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "add fn")
    fresh_reader = OnlineEngine(Settings(data_dir=tmp_path / "r2", **base), head_provider=lambda: writer.repo.head())
    assert client(fresh_reader).get("/api/status").json()["stale"] is True
    writer.sync(github=False)
    fresh_reader = OnlineEngine(Settings(data_dir=tmp_path / "r3", **base), head_provider=lambda: writer.repo.head())
    assert client(fresh_reader).get("/api/status").json()["stale"] is False
    hit = client(fresh_reader).post("/api/search", json={"question": "brand_new_fn", "mode": "bm25"}).json()["hits"][0]
    assert hit["symbol"] == "brand_new_fn"
