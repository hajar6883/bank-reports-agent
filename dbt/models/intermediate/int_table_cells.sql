-- Grain: one row per non-empty value cell of a markdown table (chunk x row x value column).
-- The first cell of each row is the row label; the following non-empty cells are the values.

with lines as (

    select
        c.chunk_id,
        c.bank,
        c.report_year,
        c.page,
        l.row_no,
        trim(l.line) as line
    from {{ ref('stg_table_chunks') }} c
    cross join lateral unnest(string_to_array(c.table_markdown, E'\n')) with ordinality as l(line, row_no)

),

rows as (

    select
        chunk_id,
        bank,
        report_year,
        page,
        row_no,
        -- "| a | b | c |" -> {a, b, c}
        string_to_array(regexp_replace(line, '^\||\|$', '', 'g'), '|') as cells
    from lines
    where line like '|%'
      and line !~ '^\|(-+\|)+$'          -- markdown separator row "|---|---|"

),

labelled as (

    select
        chunk_id,
        bank,
        report_year,
        page,
        row_no,
        trim(cells[1])                                                  as row_label,
        -- normalised label used for KPI matching: lower case, plain spaces and hyphens
        lower(trim(regexp_replace(translate(cells[1], E' ‑‐', ' --'), '\s+', ' ', 'g')))
                                                                        as row_label_clean,
        cells[2:]                                                       as value_cells
    from rows
    where cardinality(cells) >= 2

),

value_cells as (

    select
        l.*,
        v.column_index,
        trim(v.raw_cell) as raw_cell
    from labelled l
    cross join lateral unnest(l.value_cells) with ordinality as v(raw_cell, column_index)
    where trim(v.raw_cell) <> ''

)

select
    chunk_id,
    bank,
    report_year,
    page,
    row_no,
    row_label,
    row_label_clean,
    column_index,
    -- position among NON-EMPTY cells: some tables (Pillar 3 KM1) start with an empty column
    row_number() over (partition by chunk_id, row_no order by column_index) as value_position,
    count(*) over (partition by chunk_id, row_no)                           as n_filled_value_cells,
    raw_cell,
    {{ first_number_token('raw_cell') }}                     as number_token,
    {{ count_number_tokens('raw_cell', 'false') }}           as n_amount_tokens,
    {{ count_number_tokens('raw_cell', 'true') }}            as n_percent_tokens,
    {{ parse_number(first_number_token('raw_cell')) }}       as value,
    {{ first_number_token('raw_cell') }} like '%\%%'         as is_percent
from value_cells
