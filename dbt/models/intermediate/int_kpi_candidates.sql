-- Grain: one row per table cell that could be the value of a KPI (bank x fiscal year x KPI x page x row).
-- Several candidates per KPI is expected: the same figure is repeated across the report,
-- and business-line tables reuse the group's row labels. fct_bank_kpis picks one.

with kpis as (
    select * from {{ ref('kpi_definitions') }}
),

matched as (

    select
        c.bank,
        -- URD tables put the reporting year first and the comparative year second
        c.report_year - (c.value_position - 1)                            as fiscal_year,
        k.kpi_code,
        k.statement,
        case when k.report_as_positive then abs(c.value) else c.value end as value,
        c.page,
        c.chunk_id,
        c.row_no,
        c.row_label,
        c.raw_cell
    from {{ ref('int_table_cells') }} c
    join kpis k
      on c.row_label_clean ~ k.label_regex
    where c.value is not null
      and c.is_percent = k.is_percent        -- "-0.6 pt" or an amount in a ratio row is not the ratio
      and c.n_filled_value_cells <= 4        -- drop wide business-line / quarterly matrices
      and (
            c.value_position = 1
            -- comparative year, unless the cell is a merged quarterly series ("12.50% 12.46% 12.54% 12.87%"):
            -- more than 2 numbers of the KPI's own kind. "(1,530) -3.5% +2.9%*" is one amount and two changes.
            or (c.value_position = 2
                and (case when k.is_percent then c.n_percent_tokens else c.n_amount_tokens end) <= 2)
          )

),

-- Only the consolidated P&L reaches "net income attributable to equity holders":
-- business-line tables stop at pre-tax income. A page that shows it is a group P&L page.
group_pnl_pages as (
    select distinct bank, page
    from matched
    where kpi_code = 'net_income_group_share'
)

select
    m.*,
    g.page is not null as is_group_pnl_page
from matched m
left join group_pnl_pages g using (bank, page)
