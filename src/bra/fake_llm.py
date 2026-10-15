"""Deterministic stand-in for an LLM (LLM_MODEL=fake). For tests, CI and `bra demo`.

It is deliberately dumb: keyword overlap and regexes. It exists to exercise the
graph, the MCP round trip and the eval harness without an API key, not to score
well. Never report numbers produced with it.
"""
from __future__ import annotations

import re

from .models import Check, Judgement, Plan, SubQuery, Usage

STOP = set("what was the of and in a an to for at by is are did how much many compare between with from which "
           "their its on as than or vs versus ratio".split())
NUM = re.compile(r"\d[\d,.]*\s?%?")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


def _banks(system: str) -> dict[str, str]:
    return dict(re.findall(r"^- ([A-Z0-9]+): ([^,\n]+)", system, re.M))


def _mentioned(question: str, banks: dict[str, str]) -> list[str]:
    q = question.lower()
    return [code for code, name in banks.items() if code.lower() in _words(q) or name.lower() in q]


def _sources(user: str) -> list[tuple[int, str, str]]:
    blocks = re.findall(r"\[S(\d+)\] ([A-Z0-9]+) \d{4}, page \d+[^\n]*\n(.*?)(?=\n\n\[S\d+\]|\n\nQuestion:)", user, re.S)
    return [(int(n), bank, text) for n, bank, text in blocks]


def _norm_nums(text: str) -> set[str]:
    return {n.replace(",", ".").replace(" ", "").rstrip(".") for n in NUM.findall(text)}


class FakeLLM:
    name = "fake"

    async def structured(self, system: str, user: str, schema):
        u = Usage(len(system + user) // 4, 30, 1)
        if schema is Plan:
            banks = _banks(system)
            codes = _mentioned(user, banks) or [None]
            query = " ".join(sorted(_words(user)))
            return Plan(sub_queries=[SubQuery(query=query, bank=c) for c in codes]), u
        if schema is Check:
            answer = user.split("Answer to verify:\n", 1)[1]
            ok = "Not found" not in answer
            return Check(grounded=True, complete=ok, missing="" if ok else "the requested figure"), u
        if schema is Judgement:
            ref = user.split("Reference answer: ", 1)[1].split("\nCandidate answer: ", 1)[0]
            cand = user.split("\nCandidate answer: ", 1)[1]
            need = _norm_nums(ref)
            ok = bool(need) and need <= _norm_nums(cand)
            return Judgement(correct=ok, reason="all reference figures present" if ok else "figure missing"), u
        raise NotImplementedError(schema)

    async def text(self, system: str, user: str):
        question = user.rsplit("Question: ", 1)[1]
        qwords = _words(question)
        best: dict[str, tuple[int, int, str]] = {}
        for n, bank, text in _sources(user):
            for sent in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
                if not NUM.search(sent):
                    continue
                score = len(qwords & _words(sent))
                if score >= 2 and score > best.get(bank, (0, 0, ""))[0]:
                    best[bank] = (score, n, sent.strip())
        if not best:
            return "Not found in the sources.", Usage(len(user) // 4, 8, 1)
        answer = " ".join(f"{bank}: {sent} [S{n}]" for bank, (_, n, sent) in sorted(best.items()))
        return answer, Usage(len(user) // 4, len(answer) // 4, 1)
