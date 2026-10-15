"""Two ways to reach the index, behind one async interface:

- DirectSearch: calls the Store in-process (used by the baseline and the API).
- MCPSearch:    calls the MCP server's tools over stdio, exactly as an external
                client would. The agent uses this by default.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Protocol

from .config import Settings
from .embed import get_embedder
from .models import Hit
from .store import Store


class Search(Protocol):
    async def search(self, query: str, bank: str | None, year: int | None, k: int) -> list[Hit]: ...
    async def documents(self) -> list[dict]: ...


class DirectSearch:
    def __init__(self, settings: Settings):
        self.s = settings
        self.store = Store(settings.database_url, get_embedder(settings.embed_model), settings.fts_config)

    async def search(self, query, bank=None, year=None, k=5, mode="hybrid"):
        return self.store.search(query, k=k, bank=bank, year=year, mode=mode,
                                 candidates=self.s.candidates, rrf_k=self.s.rrf_k)

    async def documents(self):
        return self.store.list_documents()


def _parse_tool_output(out) -> list[dict]:
    """langchain-mcp-adapters returns a list of content blocks (one JSON object per
    list item) or a plain string, depending on version. Normalise to list[dict]."""
    if isinstance(out, tuple):  # (content, artifact) in some versions
        out = out[0]
    if isinstance(out, str):
        data = json.loads(out) if out.strip() else []
        return data if isinstance(data, list) else data.get("result", [data])
    items = []
    for block in out or []:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if text:
            data = json.loads(text)
            items.extend(data if isinstance(data, list) else [data])
    return items


class MCPSearch:
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}

    async def search(self, query, bank=None, year=None, k=5):
        args = {"query": query, "k": k}
        if bank:
            args["bank"] = bank
        if year:
            args["year"] = year
        out = await self.tools["search_reports"].ainvoke(args)
        return [Hit.from_dict(d) for d in _parse_tool_output(out)]

    async def documents(self):
        return _parse_tool_output(await self.tools["list_documents"].ainvoke({}))


@asynccontextmanager
async def mcp_search(env: dict | None = None):
    """Spawn the MCP server over stdio for the lifetime of the context."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    client = MultiServerMCPClient({
        "reports": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "bra.mcp_server"],
            "env": {**os.environ, **(env or {})},
        }
    })
    async with client.session("reports") as session:
        yield MCPSearch(await load_mcp_tools(session))
