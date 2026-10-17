from gitcontext.index.lexical import build_match, build_phrase_match, expand_identifiers, profile_query
from gitcontext.retrieval.hybrid import rrf


def test_identifier_expansion():
    assert expand_identifiers("getHTTPResponse parse_user_id") == "get http response parse user id"


def test_query_profile_extracts_exact_signals():
    p = profile_query('why does parse_user_id raise "connection refused by host"?')
    assert "parse_user_id" in p.identifiers and p.phrases == ["connection refused by host"]
    assert '"connection refused by host"' in build_phrase_match(p)
    assert '"parse"' in build_match(p)  # sub-token of snake_case identifier


def test_error_message_detected_without_quotes():
    p = profile_query("ValueError: invalid literal for int() with base 10")
    assert any("invalid literal" in ph for ph in p.phrases)


def test_rrf_prefers_agreement():
    scores, ranks = rrf({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]}, {})
    assert scores["a"] > scores["c"] and scores["b"] > scores["d"]
    assert ranks["a"] == {"dense": 1, "bm25": 2}


def test_hybrid_finds_exact_error_message(engine):
    res = engine.search('where does "payment gateway unreachable after 3 attempts" come from', top_k=3)
    top = res.hits[0].chunk
    assert top.symbol == "charge_customer"
    assert "phrase" in res.hits[0].ranks  # exact-phrase ranker contributed


def test_symbol_exact_and_metadata(engine):
    res = engine.search("refund_order", top_k=3)
    c = res.hits[0].chunk
    assert c.symbol == "refund_order" and c.author == "Alice" and c.date and c.pr_number == 1
    assert c.blob_sha and c.path == "billing/payments.py"


def test_source_and_path_filters(engine):
    res = engine.search("idempotency key charge_customer", source_types=["incident"], top_k=5)
    assert res.hits and all(h.chunk.source_type == "incident" for h in res.hits)
    res = engine.search("charge", path_prefix="web", top_k=5)
    assert all(h.chunk.path.startswith("web") for h in res.hits)


def test_all_source_types_indexed(engine):
    counts = engine.status()["chunks"]
    assert {"code", "doc", "incident", "commit"} <= set(counts)


class _Reverse:
    """Fake reranker that prefers whatever hybrid ranked last."""
    name = "reverse"

    def score(self, query, chunks):
        return [float(i) for i in range(len(chunks))]


def test_rerank_blend_controls_how_much_reranker_overrides(engine):
    q = "charge_customer retry payment gateway"
    engine.retriever.reranker = _Reverse()
    engine.s.max_chunks_per_file = 99  # disable diversity cap so we compare pure ordering
    engine.s.rerank_blend = 0.0
    base = [h.chunk.id for h in engine.search(q, rerank=False, top_k=50).hits]
    ignored = [h.chunk.id for h in engine.search(q, rerank=True, top_k=50).hits]
    assert len(base) > 2 and ignored == base  # blend 0 => reranker has no say
    engine.s.rerank_blend = 1.0
    full = [h.chunk.id for h in engine.search(q, rerank=True, top_k=50).hits]
    assert full == base[::-1]  # blend 1 => reranker fully decides
