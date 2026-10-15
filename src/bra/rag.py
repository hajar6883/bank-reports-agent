"""Baseline: single-shot RAG. Retrieve top-k for the raw question, answer once.

This is the control the agent is measured against. It shares the answer prompt
with the agent so that any difference comes from retrieval and verification.
"""
from __future__ import annotations

import re
import time

from .config import Settings
from .llm import LLM
from .models import Answer
from .prompts import ANSWER_SYSTEM, answer_user
from .retrieval import Search

_TAG = re.compile(r"\[([^\]]+)\]")
_SNUM = re.compile(r"S(\d+)")


def parse_citations(text: str) -> list[int]:
    """'... 12.9% [S2][S3] and [S1, S4]' -> [2, 3, 1, 4] (order of first use, no duplicates)."""
    seen: list[int] = []
    for tag in _TAG.findall(text):
        for n in _SNUM.findall(tag):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


async def run_rag(question: str, llm: LLM, search: Search, settings: Settings) -> Answer:
    t0 = time.perf_counter()
    hits = await search.search(question, None, None, settings.rag_top_k)
    text, usage = await llm.text(ANSWER_SYSTEM, answer_user(question, hits))
    return Answer(question=question, mode="rag", answer=text, hits=hits, context=hits,
                  cited=parse_citations(text), usage=usage, latency_s=time.perf_counter() - t0,
                  trace=[f"retrieve k={len(hits)} for the raw question"])
