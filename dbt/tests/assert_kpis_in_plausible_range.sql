-- A CET1 ratio of 98 or a negative cost/income ratio means the parser read the wrong cell.
select bank, fiscal_year, kpi_code, value, source_pages
from {{ ref('fct_bank_kpis') }}
where is_out_of_range
