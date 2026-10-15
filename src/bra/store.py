"""Postgres + pgvector store with hybrid (BM25-style + dense) retrieval.

One database does both sides of hybrid search:
- lexical: Postgres full-text search (tsvector + ts_rank_cd), OR-ed query terms
- dense:   pgvector cosine distance on an HNSW index
The two ranked lists are merged with Reciprocal Rank Fusion (RRF), which needs
no score calibration between the two retrievers.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
import yaml
from pgvector.psycopg import register_vector

from .config import ROOT
from .embed import Embedder
from .ingest import chunk_pdf, parse_filename, sha256
from .models import Hit

log = logging.getLogger(__name__)

def bank_names() -> dict[str, str]:
    """Bank code -> full name, from $BRA_BANKS_FILE or data/banks.yaml."""
    path = Path(os.environ.get("BRA_BANKS_FILE", ROOT / "data" / "banks.yaml"))
    if path.exists():
        return {k.upper(): v for k, v in (yaml.safe_load(path.read_text()) or {}).items()}
    return {}


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class Store:
    def __init__(self, dsn: str, embedder: Embedder, fts_config: str = "english"):
        self.dsn = dsn
        self.embedder = embedder
        self.fts = fts_config

    @contextmanager
    def conn(self):
        with psycopg.connect(self.dsn, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(c)
            yield c

    # ---- schema ---------------------------------------------------------
    def init(self) -> None:
        dim = self.embedder.dim
        with self.conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text NOT NULL)")
            row = c.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
            if row and row[0] != self.embedder.name:
                raise RuntimeError(
                    f"Database was built with {row[0]} but EMBED_MODEL is {self.embedder.name}. "
                    "Run `bra reset` and re-ingest, or change EMBED_MODEL back."
                )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  id serial PRIMARY KEY,
                  bank text NOT NULL,
                  year int NOT NULL,
                  path text NOT NULL,
                  sha256 text UNIQUE NOT NULL,
                  n_pages int NOT NULL,
                  ingested_at timestamptz DEFAULT now()
                )"""
            )
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                  id bigserial PRIMARY KEY,
                  doc_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                  bank text NOT NULL,
                  year int NOT NULL,
                  page int NOT NULL,
                  kind text NOT NULL,
                  text text NOT NULL,
                  header text NOT NULL,
                  embedding vector({dim}) NOT NULL,
                  tsv tsvector GENERATED ALWAYS AS
                      (to_tsvector('{self.fts}'::regconfig, header || ' ' || text)) STORED
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv)")
            c.execute("CREATE INDEX IF NOT EXISTS chunks_emb ON chunks USING hnsw (embedding vector_cosine_ops)")
            c.execute("CREATE INDEX IF NOT EXISTS chunks_bank_year ON chunks (bank, year)")
            c.execute(
                "INSERT INTO meta VALUES ('embed_model', %s) ON CONFLICT (key) DO NOTHING", (self.embedder.name,)
            )

    def reset(self) -> None:
        with self.conn() as c:
            c.execute("DROP TABLE IF EXISTS chunks, documents, meta CASCADE")

    # ---- ingestion ------------------------------------------------------
    def ingest(self, path: Path, chunk_chars: int = 1200, overlap: int = 200, tables: bool = True,
               max_pages: int | None = None) -> int:
        """Idempotent: a file whose sha256 is already stored is skipped (returns 0)."""
        bank, year = parse_filename(path)
        digest = sha256(path)
        with self.conn() as c:
            if c.execute("SELECT 1 FROM documents WHERE sha256 = %s", (digest,)).fetchone():
                log.info("skip %s (already ingested)", path.name)
                return 0
        chunks, n_pages = chunk_pdf(path, chunk_chars, overlap, tables, max_pages)
        log.info("%s: %d chunks, embedding with %s ...", path.name, len(chunks), self.embedder.name)
        name = bank_names().get(bank, bank)
        headers = [f"{name} ({bank}) {year} annual report, page {ch.page}" for ch in chunks]
        vecs = self.embedder.embed_documents([f"{h}\n{ch.text}" for h, ch in zip(headers, chunks)])
        with self.conn() as c, c.transaction():
            doc_id = c.execute(
                "INSERT INTO documents (bank, year, path, sha256, n_pages) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                (bank, year, str(path), digest, n_pages),
            ).fetchone()[0]
            with c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, bank, year, page, kind, text, header, embedding)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(doc_id, bank, year, ch.page, ch.kind, ch.text, h, np.asarray(v, dtype=np.float32))
                     for ch, h, v in zip(chunks, headers, vecs)],
                )
        log.info("ingested %s: %d pages, %d chunks", path.name, n_pages, len(chunks))
        return len(chunks)

    # ---- retrieval ------------------------------------------------------
    @staticmethod
    def _filters(bank: str | None, year: int | None) -> tuple[str, list]:
        clauses, params = [], []
        if bank:
            clauses.append("bank = %s")
            params.append(bank.upper())
        if year:
            clauses.append("year = %s")
            params.append(year)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def _dense_ids(self, c, query: str, n: int, bank, year) -> list[int]:
        where, params = self._filters(bank, year)
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        c.execute("SET hnsw.ef_search = 200")
        try:
            c.execute("SET hnsw.iterative_scan = relaxed_order")  # pgvector >= 0.8: filtered HNSW recall
        except psycopg.Error:
            pass
        rows = c.execute(
            f"SELECT id FROM chunks WHERE true{where} ORDER BY embedding <=> %s LIMIT %s", (*params, q, n)
        ).fetchall()
        return [r[0] for r in rows]

    def _lexical_ids(self, c, query: str, n: int, bank, year) -> list[int]:
        where, params = self._filters(bank, year)
        # plainto_tsquery ANDs every term, which is too strict for questions; OR them instead.
        tsq = f"to_tsquery('{self.fts}', replace(plainto_tsquery('{self.fts}', %s)::text, '&', '|'))"
        rows = c.execute(
            f"SELECT id FROM chunks WHERE tsv @@ {tsq}{where} ORDER BY ts_rank_cd(tsv, {tsq}) DESC LIMIT %s",
            (query, *params, query, n),
        ).fetchall()
        return [r[0] for r in rows]

    def search(self, query: str, k: int = 8, bank: str | None = None, year: int | None = None,
               mode: str = "hybrid", candidates: int = 50, rrf_k: int = 60) -> list[Hit]:
        with self.conn() as c:
            if mode == "dense":
                ranked = self._dense_ids(c, query, k, bank, year)
                scores = {i: 1.0 / (r + 1) for r, i in enumerate(ranked)}
            elif mode == "lexical":
                ranked = self._lexical_ids(c, query, k, bank, year)
                scores = {i: 1.0 / (r + 1) for r, i in enumerate(ranked)}
            elif mode == "hybrid":
                scores = rrf([self._dense_ids(c, query, candidates, bank, year),
                              self._lexical_ids(c, query, candidates, bank, year)], rrf_k)
                ranked = sorted(scores, key=scores.get, reverse=True)[:k]
            else:
                raise ValueError(f"mode must be dense, lexical or hybrid, not {mode}")
            if not ranked:
                return []
            rows = c.execute(
                "SELECT id, bank, year, page, kind, text FROM chunks WHERE id = ANY(%s)", (ranked,)
            ).fetchall()
        by_id = {r[0]: Hit(*r, score=scores[r[0]]) for r in rows}
        return [by_id[i] for i in ranked if i in by_id]

    def get_page(self, bank: str, year: int, page: int) -> list[Hit]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT id, bank, year, page, kind, text FROM chunks WHERE bank=%s AND year=%s AND page=%s ORDER BY id",
                (bank.upper(), year, page),
            ).fetchall()
        return [Hit(*r) for r in rows]

    def list_documents(self) -> list[dict]:
        names = bank_names()
        with self.conn() as c:
            rows = c.execute(
                "SELECT d.bank, d.year, d.n_pages, count(c.id) FROM documents d"
                " LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.bank, d.year"
            ).fetchall()
        return [{"bank": b, "name": names.get(b, b), "year": y, "pages": p, "chunks": n} for b, y, p, n in rows]
