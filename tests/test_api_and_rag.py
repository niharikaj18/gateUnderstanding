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
