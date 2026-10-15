"""bra: command line entry point.

  bra init                         create tables
  bra ingest data/raw/             index every PDF (idempotent)
  bra ask "question" [--mode rag|agent|agent_nocheck] [--direct]
  bra eval [--modes rag,agent] [--limit 5]
  bra serve                        FastAPI on :8000
  bra mcp                          MCP server on stdio
  bra reset                        drop everything
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .config import get_settings


def _tracing(settings) -> None:
    if settings.tracing.lower() != "phoenix":
        return
    try:
        from phoenix.otel import register

        register(project_name="bank-reports-agent", auto_instrument=True)
    except ImportError:
        print('TRACING=phoenix needs: pip install -e ".[tracing]"')


def _store(settings):
    from .embed import get_embedder
    from .store import Store

    return Store(settings.database_url, get_embedder(settings.embed_model), settings.fts_config)


async def _ask(args, settings) -> None:
    from .agent import run_agent
    from .llm import get_llm
    from .rag import run_rag
    from .retrieval import DirectSearch, mcp_search

    llm = get_llm(settings.llm_model)
    if args.mode == "rag":
        a = await run_rag(args.question, llm, DirectSearch(settings), settings)
    elif args.direct:
        a = await run_agent(args.question, llm, DirectSearch(settings), settings, args.mode == "agent")
    else:
        async with mcp_search() as search:
            a = await run_agent(args.question, llm, search, settings, args.mode == "agent")
    print(a.answer, "\n")
    for c in a.citations():
        print(f"  {c['source']}: {c['bank']} {c['year']} p.{c['page']}")
    print("\ntrace:")
    for t in a.trace:
        print("  -", t)
    print(f"\n{a.usage.llm_calls} LLM calls, {a.usage.input_tokens}+{a.usage.output_tokens} tokens, {a.latency_s:.1f}s")


async def _eval(args, settings) -> None:
    from .eval import run_eval
    from .llm import get_llm
    from .retrieval import DirectSearch, mcp_search

    modes = [m.strip() for m in args.modes.split(",")]
    llm, judge, direct = get_llm(settings.llm_model), get_llm(settings.judge_model), DirectSearch(settings)
    if any(m != "rag" for m in modes) and not args.direct:
        async with mcp_search() as search:
            out = await run_eval(Path(args.questions), modes, llm, judge, search, direct, settings,
                                 limit=args.limit, concurrency=args.concurrency)
    else:
        out = await run_eval(Path(args.questions), modes, llm, judge, direct, direct, settings,
                             limit=args.limit, concurrency=args.concurrency)
    print((out / "report.md").read_text())
    print(f"saved to {out}/")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bra", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("reset")
    sub.add_parser("docs")
    pi = sub.add_parser("ingest")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--max-pages", type=int, default=None, help="only the first N pages (quick tests)")
    pi.add_argument("--no-tables", action="store_true")
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--mode", default="agent", choices=["rag", "agent", "agent_nocheck"])
    pa.add_argument("--direct", action="store_true", help="agent calls the store in-process instead of via MCP")
    pe = sub.add_parser("eval")
    pe.add_argument("--questions", default="data/questions.yaml")
    pe.add_argument("--modes", default="rag,agent")
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument("--concurrency", type=int, default=2)
    pe.add_argument("--direct", action="store_true")
    ps = sub.add_parser("serve")
    ps.add_argument("--port", type=int, default=8000)
    sub.add_parser("mcp")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    _tracing(settings)

    if args.cmd == "init":
        _store(settings).init()
        print("ok")
    elif args.cmd == "reset":
        _store(settings).reset()
        print("dropped")
    elif args.cmd == "docs":
        print(json.dumps(_store(settings).list_documents(), indent=2, ensure_ascii=False))
    elif args.cmd == "ingest":
        store = _store(settings)
        store.init()
        files = []
        for raw in args.paths:
            path = Path(raw)
            files += sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        for f in files:
            n = store.ingest(f, settings.chunk_chars, settings.chunk_overlap, not args.no_tables, args.max_pages)
            print(f"{f.name}: {n} chunks" if n else f"{f.name}: already ingested, skipped")
    elif args.cmd == "ask":
        asyncio.run(_ask(args, settings))
    elif args.cmd == "eval":
        asyncio.run(_eval(args, settings))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("bra.api:app", host="127.0.0.1", port=args.port)
    elif args.cmd == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()


if __name__ == "__main__":
    main()
