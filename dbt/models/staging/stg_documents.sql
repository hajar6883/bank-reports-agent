-- One row per annual report.
select
    id          as document_id,
    upper(bank) as bank,
    year        as report_year,
    n_pages,
    ingested_at
from {{ source('bra', 'documents') }}
