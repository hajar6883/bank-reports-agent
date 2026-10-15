from __future__ import annotations

from .models import Hit

ANSWER_SYSTEM = """You answer questions about banks' annual reports using ONLY the numbered sources provided.
Rules:
- Cite every figure or claim with its source tag, e.g. [S2]. Use several tags if needed: [S1][S4].
- Give figures exactly as the source states them, with unit, scope and date (e.g. "12.9% fully loaded CET1 at 31 December 2025").
- When comparing banks, give each bank's figure with its own citation, then the comparison.
- If the sources do not contain what is asked, say "Not found in the sources" for that part. Never use outside knowledge.
- Be concise: at most 5 sentences."""

PLAN_SYSTEM = """You turn a question about banks' annual reports into search queries for a hybrid (keyword + semantic) search engine.
Indexed documents:
{documents}
Rules:
- One sub-query per bank AND per metric. "Compare X of A and B" -> one sub-query for A, one for B.
- Set `bank` to the bank's code whenever a sub-query targets one bank. Leave it null only for questions about all banks.
- Set `year` only if the question states a fiscal year.
- Phrase each query with the words a report would use (e.g. "Common Equity Tier 1 CET1 ratio 31 December"), not as a question.
- A simple single-bank, single-fact question needs exactly one sub-query. Never more than 6."""

REPLAN_SUFFIX = """
A first attempt was judged incomplete. Missing: {missing}
Already searched: {searched}
Return ONLY new sub-queries that target what is missing (different wording, or a different bank/year filter)."""

CHECK_SYSTEM = """You verify an answer written from numbered sources.
- grounded: true only if every figure and claim in the answer appears in the source it cites.
- complete: true only if every part of the question is answered (each bank, each metric). An explicit "Not found in the sources" counts as answered ONLY if the sources truly lack it.
- missing: if not grounded or not complete, say what to search for next (bank + metric). Otherwise empty."""

JUDGE_SYSTEM = """You grade a candidate answer against a reference answer for a question about bank annual reports.
correct = true only if the candidate states every fact the reference contains, with the same values (allow rounding and unit formatting such as 12.9% vs 12,9 %, EUR 9.4bn vs 9,4 Md EUR).
Extra correct detail is fine. A missing or wrong figure means correct = false."""


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        label = f"[S{i}] {h.bank} {h.year}, page {h.page}" + (" (table)" if h.kind == "table" else "")
        blocks.append(f"{label}\n{h.text}")
    return "\n\n".join(blocks)


def answer_user(question: str, hits: list[Hit]) -> str:
    return f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}"


def check_user(question: str, answer: str, hits: list[Hit]) -> str:
    return f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}\n\nAnswer to verify:\n{answer}"


def judge_user(question: str, reference: str, candidate: str) -> str:
    return f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}"
