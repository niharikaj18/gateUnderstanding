"""Synthetic evaluation set derived from the indexed repository.

Five question families, each stressing a different retriever:
  semantic     paraphrase of a docstring with the symbol name masked      (favours dense)
  exact_error  literal error message from a `raise X("...")`              (favours BM25/phrase)
  symbol       "where is `foo_bar` defined?"                              (favours BM25/symbol)
  history      a commit subject                                           (history retrieval)
  doc          a documentation heading                                    (docs retrieval)
Synthetic sets are a smoke-test for regressions, not a substitute for ~50 hand-written questions:
the JSONL schema below is what you hand-write too.

{"id": "...", "type": "...", "question": "...", "gold_keys": ["code:path::Symbol"],
 "reference_answer": "...", "must_include": ["facts", "the answer must mention"]}
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..engine import Engine
from ..index.lexical import split_identifier

RAISE_RE = re.compile(r"""raise\s+\w[\w.]*\(\s*f?(["'])(.{16,100}?)\1""")


def _first_sentence(doc: str) -> str:
    doc = " ".join(doc.strip().split())
    m = re.match(r"(.+?[.!?])(\s|$)", doc)
    return (m.group(1) if m else doc)[:220]


def _mask(sentence: str, symbol: str) -> str | None:
    name_toks = {t for t in split_identifier(symbol.split(".")[-1])} | {symbol.split(".")[-1].lower()}
    words = sentence.split()
    if sum(1 for w in words if re.sub(r"\W", "", w).lower() in name_toks) > 0 and len(name_toks) > 0:
        words = [w for w in words if re.sub(r"\W", "", w).lower() not in name_toks]
    return " ".join(words) if len(words) >= 6 else None


def generate_dataset(engine: Engine, out: Path, n: int = 60, seed: int = 7) -> int:
    rnd = random.Random(seed)
    per = max(n // 5, 1)
    code = [c for c in engine.lex.iter_chunks("code") if c.kind in ("function", "method", "class")
            and not re.search(r"(^|/)(tests?|testing)/|test_", c.path)]
    pools: dict[str, list[dict]] = {k: [] for k in ("semantic", "exact_error", "symbol", "history", "doc")}
    seen_q = set()

    for c in code:
        doc = c.extra.get("docstring", "")
        leaf = c.symbol.split(".")[-1]
        if len(doc) >= 40 and not leaf.startswith("_"):
            s = _mask(_first_sentence(doc), c.symbol)
            if s and s not in seen_q:
                seen_q.add(s)
                pools["semantic"].append(dict(type="semantic", question=f"Which code {s[0].lower() + s[1:].rstrip('.')}?",
                                              gold_keys=[c.key], reference_answer=doc[:400], must_include=[leaf]))
        if ("_" in leaf.strip("_") or re.search(r"[a-z][A-Z]", leaf)) and not leaf.startswith("__"):
            pools["symbol"].append(dict(type="symbol", question=f"Where is `{c.symbol}` defined and what does it do?",
                                        gold_keys=[c.key], reference_answer=doc[:400], must_include=[leaf]))
        for m in RAISE_RE.finditer(c.text):
            msg = max(re.split(r"\{[^}]*\}", m.group(2)), key=len).strip()
            if len(msg) >= 16 and msg not in seen_q:
                seen_q.add(msg)
                pools["exact_error"].append(dict(type="exact_error",
                                                 question=f'Where does the error message "{msg}" come from?',
                                                 gold_keys=[c.key], reference_answer="", must_include=[leaf]))
    for c in engine.lex.iter_chunks("commit"):
        subj = c.extra.get("subject", "")
        if len(subj) >= 28 and not subj.lower().startswith(("merge", "release", "bump", "update requirements")):
            pools["history"].append(dict(type="history", question=f'Which commit made this change: "{subj}"?',
                                         gold_keys=[c.key], reference_answer=subj, must_include=[c.commit_sha[:7]]))
    for c in engine.lex.iter_chunks("doc"):
        if c.kind == "section" and " > " in c.symbol and len(c.text) > 200:
            leaf = c.symbol.split(" > ")[-1]
            pools["doc"].append(dict(type="doc", question=f"What do the docs say about {leaf}?",
                                     gold_keys=[c.key], reference_answer=c.text[:400], must_include=[]))
    rows = []
    for name, pool in pools.items():
        rnd.shuffle(pool)
        rows += pool[:per]
    for i, r in enumerate(rows):
        r["id"] = f"q{i:03d}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return len(rows)
