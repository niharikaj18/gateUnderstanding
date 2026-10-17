"""Pure-function metrics (no I/O) so they are unit-testable."""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

_TOK = re.compile(r"[a-z0-9_]+")
_STOP = set("the a an of to in on for and or is are was be it this that with from by as at not no if then than so can does do".split())


def dedupe(keys: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(keys))


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    return len(g & set(dedupe(retrieved)[:k])) / len(g) if g else 0.0


def hit_at_k(retrieved, gold, k) -> float:
    return 1.0 if set(gold) & set(dedupe(retrieved)[:k]) else 0.0


def precision_at_k(retrieved, gold, k) -> float:
    top = dedupe(retrieved)[:k]
    return sum(1 for x in top if x in set(gold)) / len(top) if top else 0.0


def mrr(retrieved, gold, k=10) -> float:
    g = set(gold)
    for i, x in enumerate(dedupe(retrieved)[:k], 1):
        if x in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, gold, k) -> float:
    g = set(gold)
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(dedupe(retrieved)[:k], 1) if x in g)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(g), k) + 1))
    return dcg / idcg if idcg else 0.0


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = (len(s) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- answer-level
def _content_tokens(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP and len(t) > 2]


def token_f1(pred: str, ref: str) -> float:
    p, r = _content_tokens(pred), _content_tokens(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def key_fact_coverage(answer: str, must_include: Sequence[str]) -> float:
    if not must_include:
        return 1.0
    a = answer.lower()
    return sum(1 for m in must_include if m.lower() in a) / len(must_include)


_CITE = re.compile(r"\[(\d{1,3})\]")


def citation_metrics(answer: str, evidence_texts: list[str], evidence_keys: list[str], gold_keys: Sequence[str],
                     support_threshold: float = 0.4) -> dict:
    """
    validity   – cited numbers that point at a real evidence block / all cited numbers
    precision  – cited blocks that are gold-relevant / valid cited blocks (needs gold labels)
    support    – fraction of cited sentences whose distinctive tokens appear in the cited blocks
                 (lexical faithfulness proxy; use the LLM judge for semantic support)
    coverage   – fraction of answer sentences carrying a citation
    """
    nums = [int(n) for n in _CITE.findall(answer)]
    valid = [n for n in nums if 1 <= n <= len(evidence_texts)]
    out = {"n_citations": len(nums), "validity": (len(valid) / len(nums)) if nums else None}
    uniq = dedupe(valid)
    out["precision"] = (sum(1 for n in uniq if evidence_keys[n - 1] in set(gold_keys)) / len(uniq)) if uniq and gold_keys else None
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.strip()) > 20]
    cited_sents = [s for s in sents if _CITE.search(s)]
    sup = []
    for s in cited_sents:
        toks = set(_content_tokens(_CITE.sub("", s)))
        ev = " ".join(evidence_texts[n - 1].lower() for n in dedupe(int(x) for x in _CITE.findall(s))
                      if 1 <= n <= len(evidence_texts))
        if toks:
            sup.append(1.0 if sum(1 for t in toks if t in ev) / len(toks) >= support_threshold else 0.0)
    out["support"] = mean(sup) if sup else None
    out["coverage"] = (len(cited_sents) / len(sents)) if sents else None
    return out
