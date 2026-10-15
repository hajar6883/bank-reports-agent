"""PDF -> page-aware chunks.

Design choices (worth explaining in an interview):
- Chunks never cross a page boundary, so every citation points at one page.
- Tables are extracted separately as markdown and stored as their own chunks
  (kind="table"), because flattened table text is where annual-report RAG fails.
- Text is packed paragraph by paragraph up to ~chunk_chars with an overlap, so a
  figure and the sentence that defines it usually stay together.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pymupdf

from .models import Chunk

log = logging.getLogger(__name__)

FILENAME_RE = re.compile(r"^(?P<bank>[a-z0-9]+)_(?P<year>\d{4})", re.I)


def parse_filename(path: Path) -> tuple[str, int]:
    """data/raw/bnp_2025_urd.pdf -> ("BNP", 2025)."""
    m = FILENAME_RE.match(path.stem)
    if not m:
        raise ValueError(f"{path.name}: expected '<bank>_<year>_*.pdf', e.g. bnp_2025_urd.pdf")
    return m["bank"].upper(), int(m["year"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pack(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedy-pack paragraphs into chunks of <= max_chars, carrying `overlap`
    trailing characters into the next chunk. Oversized paragraphs are split."""
    pieces: list[str] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.find(" ") + 1:]
            cur = (tail + " " + piece).strip() if tail else piece
            if len(cur) > max_chars:  # overlap pushed it over: drop the overlap
                cur = piece
        else:
            cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def table_to_markdown(rows: list[list[str | None]]) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def chunk_pdf(path: Path, max_chars: int = 1200, overlap: int = 200, tables: bool = True,
              max_pages: int | None = None) -> tuple[list[Chunk], int]:
    doc = pymupdf.open(path)
    n_pages = doc.page_count
    out: list[Chunk] = []
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        page_no = i + 1
        if page_no % 100 == 0:
            log.info("  %s: page %d/%d", path.name, page_no, n_pages)
        text = _clean(page.get_text("text"))
        for c in pack(text.split("\n\n") if "\n\n" in text else text.split("\n"), max_chars, overlap):
            out.append(Chunk(page=page_no, kind="text", text=c))
        if tables:
            try:
                for t in page.find_tables().tables:
                    md = table_to_markdown(t.extract())
                    if md and len(md) > 40:
                        # long tables get split on row boundaries
                        for c in pack(md.split("\n"), max_chars * 2, 0):
                            out.append(Chunk(page=page_no, kind="table", text=c))
            except Exception as e:  # table detection is best effort
                log.debug("table detection failed on page %d: %s", page_no, e)
    doc.close()
    return out, n_pages
