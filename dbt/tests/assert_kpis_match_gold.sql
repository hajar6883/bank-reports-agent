-- Reconciliation: every KPI the mart produces must match the hand-labelled reference value.
-- Returns the mismatches; the test passes when it returns no rows.
select
    g.question_id,
    g.bank,
    g.fiscal_year,
    g.kpi_code,
    g.expected_value,
    f.value          as mart_value,
    f.source_pages,
    g.gold_page
from {{ ref('gold_kpis') }} g
join {{ ref('fct_bank_kpis') }} f using (bank, fiscal_year, kpi_code)
where abs(f.value - g.expected_value) > g.tolerance
