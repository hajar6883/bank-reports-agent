"""Pure unit tests: no database, no LLM."""
from pathlib import Path

import pytest

from bra.agent import interleave
from bra.eval import normalize_numbers, number_match, recall
from bra.ingest import pack, parse_filename, table_to_markdown
from bra.models import Hit
from bra.rag import parse_citations
from bra.retrieval import _parse_tool_output
from bra.store import rrf


def hit(i, bank="A", page=1):
    return Hit(chunk_id=i, bank=bank, year=2025, page=page, kind="text", text=f"t{i}")


# ---- ingestion ----------------------------------------------------------
def test_parse_filename():
    assert parse_filename(Path("data/raw/bnp_2025_urd.pdf")) == ("BNP", 2025)
    assert parse_filename(Path("CASA_2024.pdf")) == ("CASA", 2024)
    with pytest.raises(ValueError):
        parse_filename(Path("annual-report.pdf"))


def test_pack_respects_size_and_overlaps():
    paras = [("word " * 50).strip() for _ in range(10)]  # 10 x 249 chars
    chunks = pack(paras, max_chars=600, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)
    # overlap: the start of chunk n+1 repeats the tail of chunk n
    assert chunks[1].split()[0] == "word"


def test_pack_splits_oversized_paragraph():
    chunks = pack(["x" * 2500], max_chars=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_table_to_markdown():
    md = table_to_markdown([["Line", "2025"], ["Retail", "1,210"], [None, None]])
    assert md.splitlines() == ["| Line | 2025 |", "|---|---|", "| Retail | 1,210 |"]
    assert table_to_markdown([["only header"]]) == ""


# ---- retrieval ----------------------------------------------------------
def test_rrf_rewards_agreement():
    scores = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert max(scores, key=scores.get) == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert scores[3] > scores[2] and scores[3] > scores[4]


def test_interleave_gives_each_subquery_a_share():
    a = [hit(1, "A"), hit(2, "A"), hit(3, "A")]
    b = [hit(4, "B"), hit(1, "A")]  # duplicate chunk 1
    assert [h.chunk_id for h in interleave([a, b], cap=4)] == [1, 4, 2, 3]


def test_parse_tool_output_variants():
    blocks = [{"type": "text", "text": '{"chunk_id": 1}'}, {"type": "text", "text": '{"chunk_id": 2}'}]
    assert _parse_tool_output(blocks) == [{"chunk_id": 1}, {"chunk_id": 2}]
    assert _parse_tool_output('[{"chunk_id": 3}]') == [{"chunk_id": 3}]
    assert _parse_tool_output('{"result": [{"chunk_id": 4}]}') == [{"chunk_id": 4}]
    assert _parse_tool_output([]) == []


# ---- answers and metrics --------------------------------------------------
def test_parse_citations():
    assert parse_citations("A is 12.9% [S2][S3]; B is 13% [S1, S4] and again [S2].") == [2, 3, 1, 4]
    assert parse_citations("no sources") == []


@pytest.mark.parametrize("text,expected", [
    ("12,9 %", {"12.9"}), ("12.9%", {"12.9"}), ("1,210", {"1210"}),
    ("117 000", {"117000"}), ("EUR 9.4 billion", {"9.4"}), ("4.0", {"4"}),
])
def test_normalize_numbers(text, expected):
    assert normalize_numbers(text) == expected


def test_number_match():
    assert number_match("12.9%", "The CET1 ratio was 12,9 % at end-2025 [S1].")
    assert number_match("117,000 people", "employed 117 000 people")
    assert number_match("EUR 640 million.", "| Corporate banking | 640 | 702 |")  # table row
    assert not number_match("12.9% and 13.4%", "Alpha 12.9%; Beta not found.")
    assert not number_match("no figures here", "anything")  # nothing to check -> not a pass
    assert number_match("13.4% in 2025", "13.4%")  # years are ignored


def test_recall():
    gold = [{"bank": "a", "year": 2025, "page": 3}, {"bank": "B", "year": 2025, "page": 3}]
    assert recall(gold, [hit(1, "A", 3)]) == 0.5
    assert recall(gold, [hit(1, "A", 3), hit(2, "B", 3)]) == 1.0
