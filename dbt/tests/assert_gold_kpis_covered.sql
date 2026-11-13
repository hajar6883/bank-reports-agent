-- Coverage: KPIs in the gold set that the mart does not produce at all (the figure is not in a parsable table).
-- A warning, not an error: it tells us what the agent must still find in the report text.
{{ config(severity = 'warn') }}

select g.question_id, g.bank, g.fiscal_year, g.kpi_code, g.gold_page
from {{ ref('gold_kpis') }} g
left join {{ ref('fct_bank_kpis') }} f using (bank, fiscal_year, kpi_code)
where f.kpi_code is null
