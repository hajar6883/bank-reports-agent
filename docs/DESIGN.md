# Design notes

Each section is a decision, the alternative, and the reason. These are also the questions to expect about this
project in an interview.

## Chunking

**Choice:** chunks never cross a page; ~1,200 characters packed paragraph by paragraph with a 200-character overlap;
tables extracted separately (PyMuPDF `find_tables`) and stored as markdown chunks with `kind = "table"`. Each chunk is
embedded with a header ("BNP Paribas (BNP) 2025 annual report, page 123") so the bank name is in the vector even when the
passage never says it.

**Alternatives:** fixed token windows across pages (citations get vague); semantic chunking (slower, harder to
explain, little gain on structured reports); layout models for tables (better, but heavy).

**Trade-off:** small chunks retrieve precisely but lose the sentence that defines a figure; overlap mitigates that.

## Hybrid search and RRF

Dense retrieval matches meaning ("solvency ratio" ≈ "CET1"), lexical retrieval matches exact terms and figures
("CET1", "LCR", "31 December 2025"). Annual reports need both. Reciprocal Rank Fusion merges the two rankings using
ranks only, so the incomparable scores (cosine vs `ts_rank_cd`) never need calibrating. `k = 60` is the standard
constant; the ablation in the eval report shows whether hybrid beats each side on this data.

Lexical terms are OR-ed (`plainto_tsquery` ANDs them, which returns nothing for most natural-language questions).

Everything lives in one Postgres: pgvector HNSW index for vectors, GIN index for full text, a B-tree on
(bank, year) for filters. With pgvector ≥ 0.8, `hnsw.iterative_scan` keeps recall up when a filter removes most rows.

## Why an MCP server

The retrieval tools are exposed once, over a standard protocol, and any client can use them: this agent, Claude
Desktop, an IDE. The agent goes through MCP (a stdio subprocess) rather than importing the store, so the tool contract
is exercised for real. The cost is a process boundary and JSON serialisation per call, negligible next to an LLM call.

MCP vs plain function calling: function calling is how a model asks for a tool; MCP is how tools are packaged and
served so that many hosts can discover and call them without custom glue.

## Why a fixed graph rather than a ReAct agent

A ReAct loop lets the model decide every step, which is flexible but unbounded: cost and latency vary per question and
failures are hard to localise. Here the steps are fixed (plan → retrieve → answer → check) and the model decides only
what it is good at: how to split the question, and whether the answer is supported. Retries are capped
(`AGENT_MAX_ATTEMPTS`), so the worst case is known. In a bank, predictability and auditability beat autonomy.

**Where it should fail:** questions whose decomposition is not per bank (e.g. "which bank mentions X?"); the planner
falls back to an unfiltered query.

## The verifier

A second LLM call checks that each figure appears in the source it cites and that every part of the question is
answered. If not, the planner gets the "missing" note and searches again with new queries. The `agent_nocheck` mode
isolates what the verifier contributes: compare its accuracy and tokens with `agent`.

**Known weakness:** a verifier built on the same model can share its blind spots. A stricter option is a
deterministic check that every number in the answer appears verbatim in a cited chunk.

## Evaluation

- **Context recall** answers "did retrieval find it?" separately from "did the LLM use it?". Low recall with good
  answers is luck; high recall with wrong answers is a generation problem.
- **Number match** is deterministic and cannot be talked into a pass. **Judge correct** handles wording and units the
  regex cannot (EUR 9.4 billion vs 9,400 million). Report both; when they disagree, read the case.
- Use a different or stronger model as judge than as answerer, to limit self-preference bias.
- 40 questions is small: with 12 multi-bank questions, one answer is ~8 points. Report counts, not just percentages.

## What I would do next

- Rerank the fused candidates with a cross-encoder before the LLM sees them.
- Self-host an open model with vLLM and compare accuracy and cost with the API model (banks often cannot send documents
  to an external API).
- Optimise the planner and answer prompts with DSPy on a train/test split of the gold set.
- Replace the PyMuPDF table detector with a layout model, for multi-page tables and tables rendered as images.
