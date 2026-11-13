-- Grain: one row per bank x fiscal year x KPI, at group (consolidated) level.
--
-- Selection rule when candidates disagree, in order:
--   1. P&L KPIs: prefer a value read on a consolidated P&L page (one that shows net income attributable
--      to equity holders). Business-line tables reuse the same row labels but stop at pre-tax income.
--   2. the value printed on the most distinct pages.
--   3. the earliest page: group summaries come before business-line detail.
-- Every row keeps its source pages and a status, so a person can check a doubtful figure before using it.

with candidates as (
    select * from {{ ref('int_kpi_candidates') }}
),

value_support as (
    select
        bank,
        fiscal_year,
        kpi_code,
        statement,
        value,
        bool_or(is_group_pnl_page)                     as on_group_pnl_page,
        count(distinct page)                           as n_supporting_pages,
        min(page)                                      as first_page,
        array_agg(distinct page order by page)         as source_pages
    from candidates
    group by 1, 2, 3, 4, 5
),

scored as (
    select
        *,
        (statement = 'prudential' or on_group_pnl_page) as is_preferred_scope
    from value_support
),

ranked as (
    select
        *,
        row_number() over w                                        as value_rank,
        lead(is_preferred_scope) over w                            as next_is_preferred_scope,
        lead(n_supporting_pages) over w                            as next_n_supporting_pages,
        count(*) over (partition by bank, fiscal_year, kpi_code)   as n_distinct_values
    from scored
    window w as (
        partition by bank, fiscal_year, kpi_code
        order by is_preferred_scope desc, n_supporting_pages desc, first_page asc
    )
)

select
    r.bank,
    r.fiscal_year,
    r.kpi_code,
    k.kpi_name,
    k.statement,
    k.unit,
    r.value,
    r.source_pages,
    r.n_supporting_pages,
    r.n_distinct_values,
    case
        when not r.is_preferred_scope                                   then 'needs_review'  -- no group P&L source
        when r.n_distinct_values = 1                                    then 'unambiguous'
        when r.next_is_preferred_scope is distinct from true            then 'group_scope'   -- only group-level value
        when r.n_supporting_pages > r.next_n_supporting_pages           then 'majority'
        else 'needs_review'                                                                  -- tie broken by page order
    end                                                                 as selection_status,
    r.value not between k.min_value and k.max_value                     as is_out_of_range
from ranked r
join {{ ref('kpi_definitions') }} k using (kpi_code)
where r.value_rank = 1
