"""Run the evaluation: retrieval ablation + (optionally) answer correctness, citation accuracy, latency."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from ..engine import Engine
from . import metrics as M

CONFIGS = [("dense", "dense", False), ("bm25", "bm25", False), ("hybrid (RRF)", "hybrid", False),
           ("hybrid + rerank", "hybrid", True)]
KS = (1, 3, 5, 10)

JUDGE_SYSTEM = "You are a strict grader of answers about a codebase. Reply with JSON only."
JUDGE_PROMPT = """Question: {q}

Reference facts: {ref}
Must mention: {must}

Candidate answer:
{ans}

Score correctness from 0.0 to 1.0 (1 = fully correct and consistent with the reference facts, 0 = wrong or
says it cannot answer). Reply as {{"score": <float>, "reason": "<short>"}}."""


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _retrieval_pass(engine: Engine, rows: list[dict], mode: str, rerank: bool) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)
    lat: dict[str, list[float]] = defaultdict(list)
    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        res = engine.search(r["question"], mode=mode, rerank=rerank, top_k=max(KS))
        keys = [h.chunk.key for h in res.hits]
        vals = {f"recall@{k}": M.recall_at_k(keys, r["gold_keys"], k) for k in KS}
        vals.update({f"precision@{k}": M.precision_at_k(keys, r["gold_keys"], k) for k in (3, 5)})
        vals["hit@5"] = M.hit_at_k(keys, r["gold_keys"], 5)
        vals["mrr"] = M.mrr(keys, r["gold_keys"], 10)
        vals["ndcg@10"] = M.ndcg_at_k(keys, r["gold_keys"], 10)
        for k, v in vals.items():
            agg[k].append(v)
            per_type[r["type"]][k].append(v)
        for k, v in res.timings_ms.items():
            lat[k].append(v)
    return {"n": len(rows), "metrics": {k: M.mean(v) for k, v in agg.items()},
            "latency_ms": {k: {"p50": M.percentile(v, 50), "p95": M.percentile(v, 95)} for k, v in lat.items()},
            "by_type": {t: {k: M.mean(v) for k, v in d.items() if k in ("recall@5", "mrr")} for t, d in per_type.items()}}


def _judge(engine: Engine, row: dict, answer: str) -> float | None:
    if engine.llm is None:
        return None
    try:
        raw = engine.llm.complete(JUDGE_SYSTEM, JUDGE_PROMPT.format(q=row["question"], ref=row.get("reference_answer") or "n/a",
                                                                     must=", ".join(row.get("must_include", [])) or "n/a",
                                                                     ans=answer), max_tokens=200)
        return float(json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["score"])
    except Exception:
        return None


def _answer_pass(engine: Engine, rows: list[dict], k: int) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)
    e2e: list[float] = []
    for r in rows:
        t0 = time.perf_counter()
        a = engine.ask(r["question"], top_k=k)
        e2e.append((time.perf_counter() - t0) * 1000)
        ev_text = [e["text"] + " " + e["path"] + " " + e["symbol"] for e in a.evidence]
        ev_keys = [e["key"] for e in a.evidence]
        agg["key_fact_coverage"].append(M.key_fact_coverage(a.answer, r.get("must_include", [])))
        if r.get("reference_answer"):
            agg["token_f1"].append(M.token_f1(a.answer, r["reference_answer"]))
        j = _judge(engine, r, a.answer)
        if j is not None:
            agg["llm_judge_correctness"].append(j)
        cm = M.citation_metrics(a.answer, ev_text, ev_keys, r["gold_keys"])
        for name in ("validity", "precision", "support", "coverage"):
            if cm[name] is not None:
                agg[f"citation_{name}"].append(cm[name])
        agg["answered_with_any_citation"].append(1.0 if cm["n_citations"] else 0.0)
    return {"n": len(rows), "metrics": {k: M.mean(v) for k, v in agg.items()},
            "e2e_latency_ms": {"p50": M.percentile(e2e, 50), "p95": M.percentile(e2e, 95)},
            "llm": getattr(engine.llm, "name", "none (extractive; answer metrics not meaningful)")}


def run_eval(engine: Engine, dataset: Path, out_dir: Path, k: int = 5, answers: bool = False) -> str:
    rows = load(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"dataset": str(dataset), "n_questions": len(rows), "retrieval": {}}
    for name, mode, rerank in CONFIGS:
        if rerank and engine.reranker is None:
            continue
        report["retrieval"][name] = _retrieval_pass(engine, rows, mode, rerank)
    if answers:
        report["answers"] = _answer_pass(engine, rows, k)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (out_dir / "report.md").write_text(md)
    return md


def to_markdown(rep: dict) -> str:
    L = [f"## GitContext evaluation ({rep['n_questions']} questions)", "", "### Retrieval ablation", "",
         "| config | R@1 | R@3 | R@5 | R@10 | P@3 | P@5 | MRR | nDCG@10 | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rep["retrieval"].items():
        m, t = r["metrics"], r["latency_ms"]["total"]
        L.append(f"| {name} | " + " | ".join(f"{m[x]:.3f}" for x in ("recall@1", "recall@3", "recall@5", "recall@10", "precision@3", "precision@5", "mrr", "ndcg@10"))
                 + f" | {t['p50']:.0f} | {t['p95']:.0f} |")
    best = list(rep["retrieval"].values())[-1]
    L += ["", f"### By question type ({list(rep['retrieval'])[-1]}: Recall@5 / MRR)", "", "| type | R@5 | MRR |", "|---|---|---|"]
    for t, v in best["by_type"].items():
        L.append(f"| {t} | {v['recall@5']:.3f} | {v['mrr']:.3f} |")
    L += ["", "### Latency per stage (p50 / p95 ms)", "", "| config | " + " | ".join(["embed_query", "dense", "lexical", "fuse", "rerank"]) + " |", "|---|---|---|---|---|---|"]
    for name, r in rep["retrieval"].items():
        cells = []
        for st in ("embed_query", "dense", "lexical", "fuse", "rerank"):
            x = r["latency_ms"].get(st)
            cells.append(f"{x['p50']:.0f} / {x['p95']:.0f}" if x else "–")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    if "answers" in rep:
        a = rep["answers"]
        L += ["", f"### Answers ({a['llm']})", "", "| metric | value |", "|---|---|"]
        L += [f"| {k} | {v:.3f} |" for k, v in a["metrics"].items()]
        L += [f"| end-to-end latency p50 / p95 ms | {a['e2e_latency_ms']['p50']:.0f} / {a['e2e_latency_ms']['p95']:.0f} |"]
    return "\n".join(L) + "\n"
