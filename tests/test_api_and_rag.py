import hashlib
import hmac

from fastapi.testclient import TestClient

from gitcontext.api import create_app
from gitcontext.rag import generate


class FakeLLM:
    name = "fake"

    def complete(self, system, prompt, max_tokens=0):
        return "charge_customer retries three times [1]. See also the bogus ref [9]."


def test_citations_validated_against_evidence(engine):
    res = engine.search("charge_customer retry", top_k=3)
    ans = generate("q", res.hits, FakeLLM(), 20000)
    assert [c["n"] for c in ans.citations] == [1] and ans.invalid_citations == [9]
    assert ans.warnings and ans.citations[0]["path"] == "billing/payments.py"


def test_prompt_contains_provenance(engine):
    seen = {}

    class Spy(FakeLLM):
        def complete(self, system, prompt, max_tokens=0):
            seen["p"] = prompt
            return "ok [1]"

    generate("q", engine.search("refund_order", top_k=2).hits, Spy(), 20000)
    assert "author=Alice" in seen["p"] and "billing/payments.py" in seen["p"] and "date=" in seen["p"]


def test_api_end_to_end(engine):
    engine.s.webhook_secret = "s3cret"
    c = TestClient(create_app(engine=engine))
    with c:
        assert c.get("/health").json()["ok"]
        st = c.get("/status").json()
        assert st["indexed"] and not st["stale"]
        r = c.post("/query", json={"question": "where is refund_order defined?", "top_k": 4})
        assert r.status_code == 200 and r.json()["evidence"]
        r = c.post("/search", json={"question": "refund_order", "mode": "bm25"})
        assert r.json()["hits"][0]["symbol"] == "refund_order"
        assert c.post("/query", json={"question": "x"}).status_code == 422

        body = b'{"zen":"hi"}'
        bad = c.post("/webhook/github", content=body, headers={"x-github-event": "ping", "x-hub-signature-256": "sha256=0"})
        assert bad.status_code == 401
        sig = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        ok = c.post("/webhook/github", content=body, headers={"x-github-event": "ping", "x-hub-signature-256": sig})
        assert ok.json() == {"pong": True}


def test_api_key_protects_everything_except_health_and_webhook(engine):
    engine.s.api_key = "k-123"
    with TestClient(create_app(engine=engine)) as c:
        assert c.get("/health").status_code == 200  # platform health probes stay open
        assert c.get("/status").status_code == 401
        assert c.post("/query", json={"question": "refund_order"}).status_code == 401
        assert c.get("/status", headers={"x-api-key": "wrong"}).status_code == 401
        assert c.get("/status", headers={"x-api-key": "k-123"}).status_code == 200
        r = c.post("/query", json={"question": "refund_order"}, headers={"x-api-key": "k-123"})
        assert r.status_code == 200
        # webhook is gated by its own HMAC, not the API key
        assert c.post("/webhook/github", content=b"{}", headers={"x-github-event": "ping"}).status_code == 401


def test_cors_only_for_configured_origins(engine):
    engine.s.cors_origins = "https://myapp.vercel.app"
    with TestClient(create_app(engine=engine)) as c:
        ok = c.options("/query", headers={"origin": "https://myapp.vercel.app", "access-control-request-method": "POST",
                                          "access-control-request-headers": "content-type,x-api-key"})
        assert ok.headers["access-control-allow-origin"] == "https://myapp.vercel.app"
        bad = c.options("/query", headers={"origin": "https://evil.example", "access-control-request-method": "POST"})
        assert "access-control-allow-origin" not in bad.headers


def test_auto_index_on_start(repo, tmp_path):
    import time

    from gitcontext.config import Settings
    from gitcontext.engine import Engine

    s = Settings(repo=str(repo), data_dir=tmp_path / "d2", embedding_provider="hash", reranker="lexical",
                 llm_provider="none", qdrant_in_memory=True, auto_index_on_start=True, github_token=None)
    eng = Engine(s, llm=None)
    assert not eng.status()["indexed"]
    with TestClient(create_app(engine=eng)) as c:
        for _ in range(100):
            if c.get("/status").json()["indexed"]:
                break
            time.sleep(0.1)
        assert c.get("/status").json()["indexed"]
