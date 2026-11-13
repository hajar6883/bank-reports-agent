-- One row per table chunk. Text chunks and embeddings are not needed for KPI extraction.
-- Note: ingest.py splits long tables into several chunks; only the first one keeps the header row.
select
    id          as chunk_id,
    doc_id      as document_id,
    upper(bank) as bank,
    year        as report_year,
    page,
    text        as table_markdown
from {{ source('bra', 'chunks') }}
where kind = 'table'
