"""The agent: an explicit LangGraph state machine, not a free-running ReAct loop.

    plan -> retrieve -> answer -> check --(grounded & complete, or out of attempts)--> END
      ^                                  |
      +------------- replan <------------+

- plan:     the LLM decomposes the question into sub-queries with bank/year filters
            (this is where multi-bank comparisons stop failing).
- retrieve: runs every sub-query in parallel through the search tool (MCP by default)
            and interleaves results so each bank gets a share of the context.
- answer:   same prompt as the baseline, so the comparison is fair.
- check:    an LLM verifier flags unsupported or missing facts; if any, the planner
            gets the feedback and searches again. Bounded by agent_max_attempts.

Why a fixed graph: every path is bounded, each step is inspectable in a trace, and
cost per question is predictable. That matters more in a bank than open-ended autonomy.
"""
from __future__ import annotations

import asyncio
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import LLM
from .models import Answer, Check, Hit, Plan, SubQuery, Usage
from .prompts import ANSWER_SYSTEM, CHECK_SYSTEM, PLAN_SYSTEM, REPLAN_SUFFIX, answer_user, check_user
from .rag import parse_citations
from .retrieval import Search


class State(TypedDict, total=False):
    question: str
    documents: str
    batch: list[SubQuery]  # sub-queries to run next
    searched: list[str]
    groups: list[list[Hit]]  # one ranked list per sub-query run so far
    context: list[Hit]
    answer: str
    check: Check | None
    attempts: int
    usage: Annotated[list[Usage], operator.add]
    trace: Annotated[list[str], operator.add]


def interleave(groups: list[list[Hit]], cap: int) -> list[Hit]:
    """Round-robin across sub-query results, de-duplicated, capped."""
    out, seen = [], set()
    for rank in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if rank < len(g) and g[rank].chunk_id not in seen:
                seen.add(g[rank].chunk_id)
                out.append(g[rank])
                if len(out) >= cap:
                    return out
    return out


def _describe(q: SubQuery) -> str:
    return f"'{q.query}'" + (f" bank={q.bank}" if q.bank else "") + (f" year={q.year}" if q.year else "")


def build_graph(llm: LLM, search: Search, settings: Settings, self_check: bool = True):
    async def plan(state: State) -> dict:
        system = PLAN_SYSTEM.format(documents=state["documents"])
        if state.get("attempts"):
            system += REPLAN_SUFFIX.format(missing=state["check"].missing or "unspecified",
                                           searched="; ".join(state["searched"]))
        p, usage = await llm.structured(system, state["question"], Plan)
        batch = p.sub_queries[:6] or [SubQuery(query=state["question"])]
        return {"batch": batch, "usage": [usage],
                "trace": [f"plan: {len(batch)} sub-queries: " + "; ".join(_describe(q) for q in batch)]}

    async def retrieve(state: State) -> dict:
        batch = state["batch"]
        results = await asyncio.gather(*[
            search.search(q.query, q.bank, q.year, settings.agent_k_per_query) for q in batch])
        groups = state.get("groups", []) + [list(r) for r in results]
        context = interleave(groups, settings.agent_max_context)
        return {"groups": groups, "context": context,
                "searched": state.get("searched", []) + [_describe(q) for q in batch],
                "trace": [f"retrieve: {sum(len(r) for r in results)} hits, context={len(context)}"]}

    async def answer(state: State) -> dict:
        text, usage = await llm.text(ANSWER_SYSTEM, answer_user(state["question"], state["context"]))
        return {"answer": text, "usage": [usage], "trace": ["answer"]}

    async def check(state: State) -> dict:
        c, usage = await llm.structured(CHECK_SYSTEM, check_user(state["question"], state["answer"],
                                                                 state["context"]), Check)
        verdict = "pass" if (c.grounded and c.complete) else f"fail ({c.missing})"
        return {"check": c, "attempts": state.get("attempts", 0) + 1, "usage": [usage],
                "trace": [f"check: grounded={c.grounded} complete={c.complete} -> {verdict}"]}

    def route(state: State) -> str:
        c = state["check"]
        if (c.grounded and c.complete) or state["attempts"] >= settings.agent_max_attempts:
            return END
        return "plan"

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "answer")
    if self_check:
        g.add_node("check", check)
        g.add_edge("answer", "check")
        g.add_conditional_edges("check", route, ["plan", END])
    else:
        g.add_edge("answer", END)
    return g.compile()


def _documents_text(docs: list[dict]) -> str:
    return "\n".join(f"- {d['bank']}: {d['name']}, {d['year']}" for d in docs) or "- (none indexed)"


async def run_agent(question: str, llm: LLM, search: Search, settings: Settings,
                    self_check: bool = True) -> Answer:
    t0 = time.perf_counter()
    graph = build_graph(llm, search, settings, self_check)
    final = await graph.ainvoke({"question": question, "documents": _documents_text(await search.documents()),
                                 "attempts": 0, "usage": [], "trace": []})
    usage = Usage()
    for u in final["usage"]:
        usage.add(u)
    all_hits = {h.chunk_id: h for g in final["groups"] for h in g}
    return Answer(question=question, mode="agent" if self_check else "agent_nocheck",
                  answer=final["answer"], hits=list(all_hits.values()), context=final["context"],
                  cited=parse_citations(final["answer"]), usage=usage,
                  latency_s=time.perf_counter() - t0, trace=final["trace"])
