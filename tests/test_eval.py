import json

from gitcontext.evaluation import metrics as M
from gitcontext.evaluation.dataset import generate_dataset
from gitcontext.evaluation.runner import run_eval


def test_ranking_metrics():
    ret, gold = ["a", "b", "c", "d"], ["c", "z"]
    assert M.recall_at_k(ret, gold, 3) == 0.5 and M.recall_at_k(ret, gold, 2) == 0.0
    assert M.precision_at_k(ret, gold, 4) == 0.25
    assert M.mrr(ret, gold) == 1 / 3 and M.hit_at_k(ret, gold, 3) == 1.0
    assert 0 < M.ndcg_at_k(ret, gold, 4) < 1
    assert M.mrr(["a", "a", "c"], ["c"]) == 0.5  # duplicate chunks of the same key don't push rank down


def test_percentiles():
    assert M.percentile([1, 2, 3, 4, 5], 50) == 3 and M.percentile([], 95) == 0.0


def test_citation_metrics():
    ev = ["def charge_customer retries three attempts with backoff", "unrelated text about invoices"]
    ans = "The function charge_customer retries three attempts with backoff [1]. Something fabricated about kubernetes clusters [2]. Bogus [7]."
    m = M.citation_metrics(ans, ev, ["code:a::charge_customer", "code:b::x"], ["code:a::charge_customer"])
    assert m["validity"] == 2 / 3 and m["precision"] == 0.5
    assert m["support"] == 0.5  # first sentence supported, fabricated one is not


def test_answer_metrics():
    assert M.key_fact_coverage("uses ValueError and retry", ["valueerror", "retry", "x"]) == 2 / 3
    assert M.token_f1("retries payment gateway", "payment gateway retries") == 1.0


def test_dataset_and_runner(engine, tmp_path):
    n = generate_dataset(engine, tmp_path / "d.jsonl", n=10)
    assert n >= 2
    rows = [json.loads(x) for x in (tmp_path / "d.jsonl").read_text().splitlines()]
    assert all(r["gold_keys"] for r in rows)
    md = run_eval(engine, tmp_path / "d.jsonl", tmp_path / "out", answers=True)
    assert "hybrid + rerank" in md and (tmp_path / "out/report.json").exists()
