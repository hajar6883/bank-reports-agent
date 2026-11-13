# dbt: group KPIs from annual-report tables

The Python ingestion stores every table of the URDs as markdown chunks in Postgres. This dbt project turns those
tables into one tested, documented fact table of group-level KPIs, so a figure can be read from a table instead of
retrieved from PDF text, and every figure keeps the pages it came from.

```
source  public.chunks (kind = 'table')
  └─ stg_table_chunks         one row per table chunk
      └─ int_table_cells      one row per non-empty value cell (chunk x row x column)
          └─ int_kpi_candidates   every cell whose row label matches a KPI (seed: kpi_definitions)
              └─ fct_bank_kpis    one row per bank x fiscal_year x KPI
```

## Run

```bash
docker compose up -d                                  # Postgres with the ingested reports
python -m venv .venv-dbt && .venv-dbt/bin/pip install dbt-postgres
cd dbt && export DBT_PROFILES_DIR=.
../.venv-dbt/bin/dbt deps
../.venv-dbt/bin/dbt build                            # seeds, models and tests in DAG order
../.venv-dbt/bin/dbt docs generate && ../.venv-dbt/bin/dbt docs serve
```

Schemas: `staging`, `intermediate`, `marts`, `reference` (seeds).

## Grain and selection rule

`fct_bank_kpis` holds one row per bank x fiscal year x KPI, at consolidated (group) level. The same label appears in
many tables, so each KPI has several candidates. The mart picks one, in this order:

1. P&L KPIs prefer a value read on a consolidated P&L page, meaning a page that also shows net income attributable to
   equity holders. Business-line tables reuse the group's labels ("Cost/Income", "Cost of Risk") but stop at
   pre-tax income.
2. Then the value printed on the most distinct pages.
3. Then the earliest page.

`selection_status` records how each value was chosen. `needs_review` means there was no group-level source, or a tie
was broken by page order alone: a person checks it before it is used.

## Data problems handled

| Problem in the source | Where | Handling |
|---|---|---|
| Business-line tables repeat group labels, and are printed twice | BNP p156-161 and p188-192 | group P&L page rule |
| First value column empty | BPCE Pillar 3 KM1, p1036 | value position counted over non-empty cells |
| Quarterly series merged into one cell, `12.50% 12.46% 12.54% 12.87%` | Pillar 3 KM1 | not used as the comparative year |
| Value merged with its change, `(1,530) -3.5% +2.9%*` | SG p34 | first token kept; only tokens of the same kind count as a series |
| Costs printed in parentheses, `(3,350)` | all | `report_as_positive` in the KPI dictionary |
| Same name, two definitions: LCR at year end (134%) vs 12-month average (134.53%) | BNP p569 vs p351 | two KPIs, `lcr` and `lcr_avg_12m` |
| A sub-portfolio table labelled "Total risk-weighted assets" | BNP p484 | caught by the plausible-range test |

## Tests

- Keys and grain: `unique_combination_of_columns` on (bank, fiscal_year, kpi_code), `not_null`, `relationships` to
  the KPI dictionary.
- `assert_kpis_in_plausible_range`: ranges per KPI in `seeds/kpi_definitions.csv`.
- `assert_kpis_match_gold` (error): reconciles the mart with the hand-labelled values in `seeds/gold_kpis.csv`
  (from `data/questions.yaml`).
- `assert_gold_kpis_covered` (warn): gold KPIs the tables do not provide. These are figures the agent must still
  find in the report text.

Current state: 17 of 21 gold KPIs are covered and all 17 match. The 4 missing ones (year-end LCR for the three banks,
BNP ROTE) appear only in narrative text.

## Known limits

- Only the 2025 URDs are ingested, so 2024 values come from comparative columns.
- Amounts are assumed to be in EUR millions, the unit of the main tables. A table in EUR billions would pass only if
  it stayed inside the plausible range.
- Five rows are `needs_review`. One is wrong today: BPCE 2024 net income (1,712 from a sub-group page; the
  consolidated figure is 3,520 on p586).
