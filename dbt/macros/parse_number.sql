{#
  Annual-report cells are messy: "12,225", "(3,350)", "12.61%", "11,688 +4.6%" (two values merged by the PDF extractor),
  "12.50% 12.46% 12.54% 12.87%" (a quarterly series merged into one cell).
  English formatting is assumed: comma = thousands separator.
#}
{% macro number_pattern() -%}
'\(?-?[0-9][0-9,]*(?:\.[0-9]+)?\)?\s*%?'
{%- endmacro %}

{# The first numeric token of a cell. #}
{% macro first_number_token(cell) -%}
  substring({{ cell }} from {{ number_pattern() }})
{%- endmacro %}

{#
  How many numeric tokens of one kind a cell holds (percent=true: "12.5%"; percent=false: amounts like "1,530").
  "(1,530) -3.5% +2.9%*" is one amount plus two changes; "12.50% 12.46% 12.54% 12.87%" is a merged quarterly series.
#}
{% macro count_number_tokens(cell, percent) -%}
  (select count(*) from regexp_matches({{ cell }}, {{ number_pattern() }}, 'g') as m(tok)
   where (m.tok[1] like '%\%') = {{ percent }})
{%- endmacro %}

{% macro parse_number(token) -%}
  case
    when {{ token }} is null then null
    else
      (case when {{ token }} like '(%' or {{ token }} like '-%' then -1 else 1 end)
      * nullif(regexp_replace({{ token }}, '[^0-9.]', '', 'g'), '')::numeric
  end
{%- endmacro %}
