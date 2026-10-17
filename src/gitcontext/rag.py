"""Answer generation: pack retrieved evidence (with provenance) into a prompt, require [n] citations,
then validate those citations against what was actually retrieved."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .llm import LLM
from .retrieval.hybrid import Hit

SYSTEM = """You are GitContext, a senior engineer who helps people understand an unfamiliar codebase.
Answer ONLY from the numbered evidence blocks provided. Rules:
- Cite every factual claim with its evidence number in square brackets, e.g. [1] or [2][4]. Never cite a number that was not provided.
- Use code evidence for "how/where" questions; commits, PRs and incident reports for "why/when/who" questions.
- Name concrete files, functions and classes. Quote short snippets only when exact wording matters (error text, signatures).
- If the evidence does not contain the answer, say so plainly and state what is missing. Do not guess or invent APIs.
- Each block shows its last-modified date; if evidence looks old or contradicts newer evidence, say so.
Be concise: lead with the direct answer, then supporting detail."""

CITE_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    index_status: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    model: str = ""

    def to_dict(self):
        return self.__dict__.copy()


def format_evidence(hits: list[Hit], char_budget: int) -> tuple[str, list[Hit]]:
    blocks, used, total = [], [], 0
    for h in hits:
        c = h.chunk
        meta = [f"type={c.source_type}", f"ref={c.label()}"]
        if c.symbol and c.source_type in ("code", "doc", "incident"):
            meta.append(f"symbol={c.symbol}")
        if c.author:
            meta.append(f"author={c.author}")
        if c.date:
            meta.append(f"date={c.date[:10]}")
        if c.commit_sha and c.source_type != "commit":
            meta.append(f"last_commit={c.commit_sha[:7]}")
        if c.pr_number and c.source_type not in ("pr", "issue", "incident"):
            meta.append(f"PR=#{c.pr_number}")
        body = c.text if len(c.text) <= 3500 else c.text[:3500] + "\n...[truncated]"
        block = f"[{len(used) + 1}] " + " | ".join(meta) + f"\n```\n{body}\n```"
        if total + len(block) > char_budget and used:
            break
        blocks.append(block)
        used.append(h)
        total += len(block)
    return "\n\n".join(blocks), used


def extractive_answer(question: str, hits: list[Hit]) -> str:
    if not hits:
        return "No relevant evidence found in the index."
    lines = ["_No LLM configured — showing the most relevant evidence (set ANTHROPIC_API_KEY for synthesized answers)._", ""]
    for i, h in enumerate(hits[:3], 1):
        c = h.chunk
        snippet = " ".join(c.text.split())[:280]
        lines.append(f"- **{c.label()}**" + (f" (`{c.symbol}`)" if c.symbol else "") + f": {snippet}… [{i}]")
    return "\n".join(lines)


def generate(question: str, hits: list[Hit], llm: LLM | None, char_budget: int) -> Answer:
    t0 = time.perf_counter()
    ctx, used = format_evidence(hits, char_budget)
    if llm is None:
        text, model = extractive_answer(question, used), "extractive"
    else:
        prompt = f"Question: {question}\n\nEvidence:\n\n{ctx}\n\nAnswer with citations:"
        text, model = llm.complete(SYSTEM, prompt), llm.name
    nums = sorted({int(n) for n in CITE_RE.findall(text)})
    valid = [n for n in nums if 1 <= n <= len(used)]
    invalid = [n for n in nums if n not in valid]
    citations = []
    for n in valid:
        c = used[n - 1].chunk
        citations.append({"n": n, "id": c.id, "key": c.key, "label": c.label(), "source_type": c.source_type,
                          "path": c.path, "symbol": c.symbol, "start_line": c.start_line, "end_line": c.end_line,
                          "author": c.author, "date": c.date, "commit_sha": c.commit_sha,
                          "pr_number": c.pr_number, "url": c.url})
    ans = Answer(question, text, citations, invalid, [h.to_dict() for h in used], model=model)
    ans.timings_ms["generate"] = (time.perf_counter() - t0) * 1000
    if invalid:
        ans.warnings.append(f"Model cited evidence numbers that were not provided: {invalid}")
    return ans
