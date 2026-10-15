"""Agent graph behaviour with a scripted LLM and an in-memory search (no DB)."""
from bra.agent import run_agent
from bra.config import Settings
from bra.fake_llm import FakeLLM
from bra.models import Check, Hit, Usage

DOCS = [{"bank": "ALPHA", "name": "Banque Alpha", "year": 2025}, {"bank": "BETA", "name": "Beta Bank", "year": 2025}]
TEXT = {
    "ALPHA": "The CET1 ratio of Banque Alpha stood at 12.9% at 31 December 2025.",
    "BETA": "Beta Bank's CET1 ratio reached 13.4% at end-December 2025.",
}


class MemorySearch:
    def __init__(self):
        self.calls = []

    async def search(self, query, bank=None, year=None, k=5):
        self.calls.append((query, bank))
        banks = [bank] if bank else list(TEXT)
        return [Hit(chunk_id=i, bank=b, year=2025, page=3, kind="text", text=TEXT[b])
                for i, b in enumerate(banks, start=1 if bank != "BETA" else 2)]

    async def documents(self):
        return DOCS


class NeverSatisfied(FakeLLM):
    """Verifier always says something is missing: the loop must still stop."""

    async def structured(self, system, user, schema):
        if schema is Check:
            return Check(grounded=False, complete=False, missing="everything"), Usage(1, 1, 1)
        return await super().structured(system, user, schema)


async def test_comparison_is_decomposed_per_bank():
    search = MemorySearch()
    a = await run_agent("Compare the CET1 ratio of Banque Alpha and Beta Bank", FakeLLM(), search, Settings())
    assert sorted(b for _, b in search.calls) == ["ALPHA", "BETA"]
    assert "12.9%" in a.answer and "13.4%" in a.answer
    assert {c["bank"] for c in a.citations()} == {"ALPHA", "BETA"}
    assert a.trace[-1].startswith("check: grounded=True complete=True")
    assert a.usage.llm_calls == 3  # plan, answer, check


async def test_retry_loop_is_bounded():
    settings = Settings(agent_max_attempts=2)
    a = await run_agent("CET1 ratio of Beta Bank", NeverSatisfied(), MemorySearch(), settings)
    assert sum(t.startswith("check") for t in a.trace) == 2
    assert sum(t.startswith("plan") for t in a.trace) == 2
    assert a.usage.llm_calls == 6


async def test_nocheck_mode_skips_verifier():
    a = await run_agent("CET1 ratio of Beta Bank", FakeLLM(), MemorySearch(), Settings(), self_check=False)
    assert a.mode == "agent_nocheck"
    assert not any(t.startswith("check") for t in a.trace)
    assert a.usage.llm_calls == 2
