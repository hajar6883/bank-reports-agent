"""HTTP API.  uvicorn bra.api:app  (or `bra serve`)

The agent reaches the index through one MCP session opened for the app's
lifetime, the same path an external MCP client takes. Set RETRIEVER=direct to
call the store in-process instead.
"""
from __future__ import annotations

import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import run_agent
from .config import get_settings
from .llm import get_llm
from .rag import run_rag
from .retrieval import DirectSearch, mcp_search

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state["settings"] = settings
    state["llm"] = get_llm(settings.llm_model)
    state["direct"] = DirectSearch(settings)
    async with AsyncExitStack() as stack:
        if os.environ.get("RETRIEVER", "mcp") == "mcp":
            state["agent_search"] = await stack.enter_async_context(mcp_search())
        else:
            state["agent_search"] = state["direct"]
        yield
    state.clear()


app = FastAPI(title="bank-reports-agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["rag", "agent", "agent_nocheck"] = "agent"


class AskResponse(BaseModel):
    answer: str
    mode: str
    citations: list[dict]
    trace: list[str]
    input_tokens: int
    output_tokens: int
    latency_s: float


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "documents": await state["direct"].documents()}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    s, llm = state["settings"], state["llm"]
    try:
        if req.mode == "rag":
            a = await run_rag(req.question, llm, state["direct"], s)
        else:
            a = await run_agent(req.question, llm, state["agent_search"], s, self_check=req.mode == "agent")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"pipeline error: {e}") from e
    return AskResponse(answer=a.answer, mode=a.mode, citations=a.citations(), trace=a.trace,
                       input_tokens=a.usage.input_tokens, output_tokens=a.usage.output_tokens,
                       latency_s=round(a.latency_s, 2))
