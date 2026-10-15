"""MCP server exposing the annual-report index as tools.

Run over stdio:   python -m bra.mcp_server
Any MCP client can use it: the LangGraph agent in this repo, Claude Desktop,
Cursor, etc. (see README for the client config snippet).
"""
from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .embed import get_embedder
from .store import Store

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)  # stdout is the MCP channel

mcp = FastMCP(
    "bank-reports",
    instructions=(
        "Search the annual reports (universal registration documents) of French banks. "
        "Call list_documents first to see which banks and years are indexed, then search_reports "
        "with a bank filter for each bank you need. Cite results as BANK YEAR p.PAGE."
    ),
)

_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        s = get_settings()
        _store = Store(s.database_url, get_embedder(s.embed_model), s.fts_config)
    return _store


@mcp.tool()
def list_documents() -> list[dict]:
    """List the indexed reports: bank code, full name, year, page and chunk counts."""
    return store().list_documents()


@mcp.tool()
def search_reports(query: str, bank: str | None = None, year: int | None = None, k: int = 5,
                   mode: str = "hybrid") -> list[dict]:
    """Search the reports. Returns up to k passages with bank, year, page and text.

    Args:
        query: what to look for, phrased like report text (e.g. "CET1 ratio 31 December 2025").
        bank: optional bank code from list_documents (e.g. "BNP"); omit to search all banks.
        year: optional fiscal year.
        k: number of passages (1-20).
        mode: "hybrid" (default), "dense" or "lexical".
    """
    s = get_settings()
    hits = store().search(query, k=max(1, min(k, 20)), bank=bank, year=year, mode=mode,
                          candidates=s.candidates, rrf_k=s.rrf_k)
    return [h.to_dict() for h in hits]


@mcp.tool()
def get_page(bank: str, year: int, page: int) -> list[dict]:
    """Return every chunk (text and tables) of one report page, to read a table or check context."""
    return [h.to_dict() for h in store().get_page(bank, year, page)]


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
