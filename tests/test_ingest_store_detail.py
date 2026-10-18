"""Detailed tests for ingest.py and store.py.

Unit tests (no marker) run anywhere. Tests marked `db` need Postgres on :5433
(`docker compose up -d`) and are skipped otherwise.
"""
from pathlib import Path

import pymupdf
import pytest

from bra.ingest import _clean, chunk_pdf, pack, sha256, table_to_markdown
from bra.store import Store, rrf


# ==== ingest.py: _clean ===================================================
def test_clean_removes_soft_hyphens():
    assert _clean("capi­tal") == "capital"


def test_clean_joins_words_split_across_lines_only_before_lowercase():
    assert _clean("capi-\ntal") == "capital"
    assert _clean("CET1-\nRatio") == "CET1-\nRatio"  # uppercase next: keep the hyphen


def test_clean_collapses_spaces_and_blank_lines():
    assert _clean("a  \t b\n\n\n\nc") == "a b\n\nc"
    assert _clean("   padded   ") == "padded"


# ==== ingest.py: pack =====================================================
PARAS = [
    "Capital ratios are computed under Basel III.",
    "The CET1 ratio stood at 12.9% at end-2025.",
    "Net income rose to EUR 11.7bn.",
]


def test_pack_exact_output_with_overlap():
    assert pack(PARAS, max_chars=60, overlap=20) == [
        "Capital ratios are computed under Basel III.",
        "under Basel III. The CET1 ratio stood at 12.9% at end-2025.",
        "12.9% at end-2025. Net income rose to EUR 11.7bn.",
    ]


def test_pack_without_overlap_does_not_repeat_text():
    chunks = pack(PARAS, max_chars=60, overlap=0)
    assert chunks == PARAS


def test_pack_joins_small_paragraphs_with_newline():
    assert pack(["a", "b", "c"], max_chars=100, overlap=10) == ["a\nb\nc"]


def test_pack_skips_empty_paragraphs():
    assert pack(["", "   ", "x"], max_chars=10, overlap=0) == ["x"]
    assert pack([], max_chars=10, overlap=0) == []


def test_pack_overlap_starts_on_a_word_boundary():
    chunks = pack(["alpha beta gamma delta", "epsilon"], max_chars=25, overlap=8)
    # tail of chunk 1 is "ma delta" -> trimmed to "delta", never a half word like "ma"
    assert chunks[1] == "delta epsilon"


def test_pack_drops_overlap_when_it_would_exceed_the_limit():
    chunks = pack(["aaaa bbbb", "c" * 20], max_chars=20, overlap=9)
    assert chunks == ["aaaa bbbb", "c" * 20]


def test_pack_cuts_oversized_paragraph_at_a_space():
    chunks = pack(["word " * 30], max_chars=50, overlap=0)
    assert all(len(c) <= 50 for c in chunks)
    assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)
    assert " ".join(chunks).split() == ["word"] * 30  # nothing lost


# ==== ingest.py: table_to_markdown ========================================
def test_table_to_markdown_pads_ragged_rows_and_flattens_cell_newlines():
    md = table_to_markdown([["Line", "2025", "2024"], ["Retail\nbanking", "1,210"]])
    assert md.splitlines() == [
        "| Line | 2025 | 2024 |",
        "|---|---|---|",
        "| Retail banking | 1,210 |  |",
    ]


# ==== ingest.py: sha256 ===================================================
def test_sha256_is_stable_and_content_based(tmp_path):
    a, b, c = tmp_path / "a.bin", tmp_path / "b.bin", tmp_path / "c.bin"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"different")
    assert sha256(a) == sha256(b)  # name doesn't matter
    assert sha256(a) != sha256(c)
    assert len(sha256(a)) == 64


# ==== ingest.py: chunk_pdf (on the synthetic corpus) ======================
@pytest.fixture(scope="module")
def alpha_pdf(corpus) -> Path:
    return next(p for p in corpus if p.name.startswith("alpha"))


def test_chunk_pdf_pages_are_1_based(alpha_pdf):
    chunks, n_pages = chunk_pdf(alpha_pdf)
    assert n_pages == 6
    assert {c.page for c in chunks} == {1, 2, 3, 4, 5, 6}


def test_chunk_pdf_figure_lands_on_the_right_page(alpha_pdf):
    chunks, _ = chunk_pdf(alpha_pdf)
    pages = {c.page for c in chunks if "12.9%" in c.text}
    assert pages == {3}


def test_chunk_pdf_extracts_table_as_markdown(alpha_pdf):
    chunks, _ = chunk_pdf(alpha_pdf)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1 and tables[0].page == 5
    assert "| Corporate banking | 640 | 702 |" in tables[0].text


def test_chunk_pdf_tables_flag_and_max_pages(alpha_pdf):
    no_tables, _ = chunk_pdf(alpha_pdf, tables=False)
    assert all(c.kind == "text" for c in no_tables)
    first_two, n_pages = chunk_pdf(alpha_pdf, max_pages=2)
    assert {c.page for c in first_two} == {1, 2}
    assert n_pages == 6  # still reports the real page count


def test_chunk_pdf_respects_chunk_size(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), "Lorem ipsum dolor sit amet. " * 80, fontsize=8)
    path = tmp_path / "x_2025.pdf"
    doc.save(path)
    chunks, _ = chunk_pdf(path, max_chars=300, overlap=50)
    assert len(chunks) > 1 and all(len(c.text) <= 300 for c in chunks)


# ==== store.py: rrf =======================================================
def test_rrf_exact_scores():
    s = rrf([["A", "B", "C"], ["C", "A", "D"]], k=60)
    assert s["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["D"] == pytest.approx(1 / 63)
    assert sorted(s, key=s.get, reverse=True) == ["A", "C", "B", "D"]


# ==== store.py: _filters ==================================================
def test_filters_builds_sql_and_uppercases_bank():
    assert Store._filters(None, None) == ("", [])
    assert Store._filters("bnp", None) == (" AND bank = %s", ["BNP"])
    assert Store._filters("bnp", 2025) == (" AND bank = %s AND year = %s", ["BNP", 2025])


# ==== store.py: against Postgres ==========================================
@pytest.mark.db
def test_init_refuses_a_different_embedding_model(store, settings):
    from bra.embed import get_embedder

    other = Store(settings.database_url, get_embedder("hash:128"))
    with pytest.raises(RuntimeError, match="hash:256"):
        other.init()


@pytest.mark.db
def test_header_is_stored_and_searchable(store):
    with store.conn() as c:
        header = c.execute(
            "SELECT header FROM chunks WHERE bank = 'BETA' AND page = 3 LIMIT 1"
        ).fetchone()[0]
    assert header == "Beta Bank (BETA) 2025 annual report, page 3"
    # "annual report" only appears in headers, never in the page text
    assert store.search("annual report", k=3, mode="lexical", bank="BETA")


@pytest.mark.db
def test_lexical_query_terms_are_ored(store):
    # an unknown word must not stop the known one from matching
    hits = store.search("CET1 zzzunknownword", k=3, mode="lexical")
    assert hits and all("CET1" in h.text for h in hits)


@pytest.mark.db
@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_search_returns_at_most_k_sorted_by_score(store, mode):
    hits = store.search("net income", k=3, mode=mode)
    assert 0 < len(hits) <= 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.db
def test_year_filter_and_bad_mode(store):
    assert store.search("CET1", year=1999) == []
    with pytest.raises(ValueError):
        store.search("CET1", mode="bm25")


@pytest.mark.db
def test_get_page_is_case_insensitive_and_complete(store):
    hits = store.get_page("alpha", 2025, 5)
    assert hits and {h.page for h in hits} == {5}
    assert [h.chunk_id for h in hits] == sorted(h.chunk_id for h in hits)
    assert store.get_page("ALPHA", 2025, 99) == []
