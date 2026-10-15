"""Thin LLM interface so the pipelines can run against any provider or a fake.

`structured` returns a validated Pydantic object; `text` returns a string. Both
return token usage so the eval can report cost per question.
"""
from __future__ import annotations

from typing import Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .models import Usage

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    name: str

    async def structured(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]: ...
    async def text(self, system: str, user: str) -> tuple[str, Usage]: ...


def _usage(msg) -> Usage:
    meta = getattr(msg, "usage_metadata", None) or {}
    return Usage(meta.get("input_tokens", 0), meta.get("output_tokens", 0), 1)


class LangChainLLM:
    def __init__(self, spec: str, temperature: float = 0.0):
        from langchain.chat_models import init_chat_model

        self.name = spec
        self.model = init_chat_model(spec, temperature=temperature)

    async def structured(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        runnable = self.model.with_structured_output(schema, include_raw=True)
        out = await runnable.ainvoke([SystemMessage(system), HumanMessage(user)])
        if out.get("parsed") is None:
            raise ValueError(f"{self.name} returned unparseable {schema.__name__}: {out.get('parsing_error')}")
        return out["parsed"], _usage(out["raw"])

    async def text(self, system: str, user: str) -> tuple[str, Usage]:
        msg = await self.model.ainvoke([SystemMessage(system), HumanMessage(user)])
        content = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content if isinstance(b, dict))
        return content.strip(), _usage(msg)


def get_llm(spec: str) -> LLM:
    if spec.startswith("fake"):
        from .fake_llm import FakeLLM

        return FakeLLM()
    return LangChainLLM(spec)
