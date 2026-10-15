"""Against a real Postgres + pgvector (DATABASE_URL), with the synthetic corpus."""
import pytest

pytestmark = pytest.mark.db


def test_ingest_is_idempotent(store, corpus):
    assert all(store.ingest(p) == 0 for p in corpus)
    docs = store.list_documents()
    assert [(d["bank"], d["pages"]) for d in docs] == [("ALPHA", 6), ("BETA", 4)]
    assert docs[0]["name"] == "Banque Alpha"


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_search_modes_find_the_page(store, mode):
    hits = store.search("CET1 ratio Beta Bank", k=3, mode=mode, bank="BETA")
    assert hits and (hits[0].bank, hits[0].page) == ("BETA", 3)


def test_bank_filter(store):
    hits = store.search("CET1 ratio", k=10, bank="alpha")
    assert hits and {h.bank for h in hits} == {"ALPHA"}


def test_tables_are_separate_chunks(store):
    kinds = {h.kind for h in store.get_page("ALPHA", 2025, 5)}
    assert kinds == {"text", "table"}
    hits = store.search("cost of risk corporate banking", k=5, bank="ALPHA")
    assert any(h.kind == "table" and "| Corporate banking | 640 |" in h.text for h in hits)


async def test_mcp_round_trip(store):
    from bra.retrieval import mcp_search

    async with mcp_search() as search:
        assert set(search.tools) == {"list_documents", "search_reports", "get_page"}
        docs = await search.documents()
        assert {d["bank"] for d in docs} == {"ALPHA", "BETA"}
        hits = await search.search("CET1 ratio", "BETA", 2025, 2)
        assert hits[0].bank == "BETA" and hits[0].page == 3


async def test_eval_end_to_end(store, settings, tmp_path):
    from pathlib import Path

    from bra.eval import run_eval
    from bra.fake_llm import FakeLLM
    from bra.retrieval import DirectSearch

    direct = DirectSearch(settings)
    out = await run_eval(Path(__file__).parent / "fixtures" / "questions.yaml", ["rag", "agent"], FakeLLM(),
                         FakeLLM(), direct, direct, settings, out_root=tmp_path)
    report = (out / "report.md").read_text()
    assert "| agent/multi |" in report and "## Retrieval only" in report
    assert len((out / "results.jsonl").read_text().splitlines()) == 12


def test_api(store, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RETRIEVER", "direct")
    from bra.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/ask", json={"question": "Compare the CET1 ratio of Banque Alpha and Beta Bank",
                                      "mode": "agent"})
        assert r.status_code == 200
        body = r.json()
        assert {c["bank"] for c in body["citations"]} == {"ALPHA", "BETA"}
        assert client.post("/ask", json={"question": "x"}).status_code == 422
