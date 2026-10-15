"""Evaluation harness.

For every question in the gold set and every mode (rag, agent, agent_nocheck):
- context_recall: share of gold (bank, year, page) sources present in what the LLM saw
- number_match:   every number in the reference answer appears in the answer (deterministic)
- judge_correct:  LLM judge against the reference answer (catches wording/unit variants)
- cites_gold:     at least one citation points at a gold page
- tokens, LLM calls, latency, cost (if prices are set)
Plus a retrieval-only ablation (dense vs lexical vs hybrid recall@k on the raw
question) that costs no LLM calls.

Gold set format: see data/questions.yaml.
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from .agent import run_agent
from .config import Settings
from .llm import LLM
from .models import Answer, Hit, Judgement
from .prompts import JUDGE_SYSTEM, judge_user
from .rag import run_rag
from .retrieval import DirectSearch, Search

_NUM = re.compile(r"\d[\d\s  .,]*\d|\d")


def _norm(raw: str) -> str:
    s = re.sub(r"[\s\u00a0\u202f]", "", raw)
    s = re.sub(r",(?=\d{3}(\D|$))", "", s)  # 1,210 -> 1210 (thousands)
    s = s.replace(",", ".")  # 12,9 -> 12.9 (decimal)
    if s.count(".") > 1:  # 1.210.000 style thousands
        s = s.replace(".", "")
    s = s.rstrip(".")
    return s[:-2] if s.endswith(".0") else s


def normalize_numbers(text: str, lenient: bool = False) -> set[str]:
    """'12,9 %' '12.9%' -> '12.9';  '1,210' '1 210' -> '1210';  '117 000' -> '117000'.

    A space can be a French thousands separator ('117 000') or a gap between two
    table cells ('640 702'). With lenient=True every space-separated piece is
    added too, so table rows in an answer still match. Use strict for the
    reference and lenient for the candidate."""
    out = set()
    for raw in _NUM.findall(text):
        out.add(_norm(raw))
        if lenient:
            out.update(_norm(p) for p in re.split(r"[\s\u00a0\u202f]+", raw) if p)
    return out


def number_match(reference: str, answer: str) -> bool:
    ref = {n for n in normalize_numbers(reference) if not re.fullmatch(r"(19|20)\d\d", n)}  # ignore years
    return bool(ref) and ref <= normalize_numbers(answer, lenient=True)


def recall(gold: list[dict], hits: list[Hit]) -> float:
    if not gold:
        return float("nan")
    got = {(h.bank, h.year, h.page) for h in hits}
    return sum((g["bank"].upper(), int(g["year"]), int(g["page"])) in got for g in gold) / len(gold)


def load_questions(path: Path) -> list[dict]:
    qs = yaml.safe_load(path.read_text())["questions"]
    ready = [q for q in qs if str(q.get("answer", "")).strip() and q.get("sources")]
    skipped = len(qs) - len(ready)
    if skipped:
        print(f"note: {skipped} question(s) skipped (no reference answer or sources yet)")
    return ready


async def _run_one(q: dict, mode: str, llm: LLM, judge: LLM, agent_search: Search, direct: Search,
                   settings: Settings) -> dict:
    try:
        if mode == "rag":
            ans: Answer = await run_rag(q["question"], llm, direct, settings)
        else:
            ans = await run_agent(q["question"], llm, agent_search, settings, self_check=(mode == "agent"))
    except Exception as e:  # one failure should not kill a 40-question run
        return {"id": q["id"], "type": q.get("type", "single"), "mode": mode, "error": repr(e)}
    j, ju = await judge.structured(JUDGE_SYSTEM, judge_user(q["question"], str(q["answer"]), ans.answer), Judgement)
    gold = {(s["bank"].upper(), int(s["year"]), int(s["page"])) for s in q["sources"]}
    cited = {(c["bank"], c["year"], c["page"]) for c in ans.citations()}
    cost = (ans.usage.input_tokens * settings.price_input_per_m
            + ans.usage.output_tokens * settings.price_output_per_m) / 1e6
    return {
        "id": q["id"], "type": q.get("type", "single"), "mode": mode, "question": q["question"],
        "reference": q["answer"], "answer": ans.answer,
        "context_recall": recall(q["sources"], ans.context),
        "retrieved_recall": recall(q["sources"], ans.hits),
        "number_match": number_match(str(q["answer"]), ans.answer),
        "judge_correct": j.correct, "judge_reason": j.reason,
        "cites_gold": bool(cited & gold),
        "input_tokens": ans.usage.input_tokens, "output_tokens": ans.usage.output_tokens,
        "llm_calls": ans.usage.llm_calls, "latency_s": round(ans.latency_s, 2), "cost": cost,
        "citations": ans.citations(), "trace": ans.trace,
    }


async def retrieval_ablation(questions: list[dict], direct: DirectSearch, k: int) -> list[dict]:
    rows = []
    for q in questions:
        row = {"id": q["id"], "type": q.get("type", "single")}
        for mode in ("dense", "lexical", "hybrid"):
            row[mode] = recall(q["sources"], await direct.search(q["question"], k=k, mode=mode))
        rows.append(row)
    return rows


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x == x]  # drop NaN
    return statistics.mean(xs) if xs else float("nan")


def summarize(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        groups[(r["mode"], "all")].append(r)
        groups[(r["mode"], r["type"])].append(r)
    summary = {}
    for (mode, typ), rs in sorted(groups.items()):
        summary[f"{mode}/{typ}"] = {
            "n": len(rs),
            "context_recall": _mean(r["context_recall"] for r in rs),
            "number_match": _mean(r["number_match"] for r in rs),
            "judge_correct": _mean(r["judge_correct"] for r in rs),
            "cites_gold": _mean(r["cites_gold"] for r in rs),
            "tokens_per_q": _mean(r["input_tokens"] + r["output_tokens"] for r in rs),
            "llm_calls_per_q": _mean(r["llm_calls"] for r in rs),
            "latency_s": _mean(r["latency_s"] for r in rs),
            "cost_per_1k_q": 1000 * _mean(r["cost"] for r in rs),
        }
    summary["errors"] = sum("error" in r for r in results)
    return summary


def report_md(summary: dict, ablation: list[dict], meta: dict) -> str:
    pct = lambda x: "n/a" if x != x else f"{100 * x:.0f}%"  # noqa: E731
    lines = [f"# Evaluation report ({meta['timestamp']})", "",
             f"LLM `{meta['llm']}` · judge `{meta['judge']}` · embeddings `{meta['embed']}` · "
             f"{meta['n_questions']} questions · errors: {summary['errors']}", "",
             "## Answers", "",
             "| mode / type | n | context recall | number match | judge correct | cites gold page | tokens/q | LLM calls/q | latency (s) | cost / 1k q |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, s in summary.items():
        if key == "errors":
            continue
        cost = "n/a" if not meta["priced"] else f"{s['cost_per_1k_q']:.2f}"
        lines.append(f"| {key} | {s['n']} | {pct(s['context_recall'])} | {pct(s['number_match'])} | "
                     f"{pct(s['judge_correct'])} | {pct(s['cites_gold'])} | {s['tokens_per_q']:.0f} | "
                     f"{s['llm_calls_per_q']:.1f} | {s['latency_s']:.1f} | {cost} |")
    if ablation:
        lines += ["", f"## Retrieval only (raw question, recall@{meta['k']})", "",
                  "| type | n | dense | lexical | hybrid |", "|---|---|---|---|---|"]
        by_type = defaultdict(list)
        for r in ablation:
            by_type["all"].append(r)
            by_type[r["type"]].append(r)
        for t, rs in by_type.items():
            lines.append(f"| {t} | {len(rs)} | " + " | ".join(pct(_mean(r[m] for r in rs))
                                                            for m in ("dense", "lexical", "hybrid")) + " |")
    lines += ["", "Per-question details, answers and agent traces: `results.jsonl`."]
    return "\n".join(lines) + "\n"


async def run_eval(questions_path: Path, modes: list[str], llm: LLM, judge: LLM, agent_search: Search,
                   direct: DirectSearch, settings: Settings, out_root: Path = Path("results"),
                   limit: int | None = None, concurrency: int = 2) -> Path:
    questions = load_questions(questions_path)[:limit]
    if not questions:
        raise SystemExit("No question has both a reference answer and sources yet. Fill data/questions.yaml.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def guarded(q, m):
        async with sem:
            r = await _run_one(q, m, llm, judge, agent_search, direct, settings)
            mark = "x" if "error" in r else ("+" if r["judge_correct"] else "-")
            print(f"[{mark}] {m:13s} {q['id']}")
            return r

    t0 = time.perf_counter()
    results = await asyncio.gather(*[guarded(q, m) for q in questions for m in modes])
    ablation = await retrieval_ablation(questions, direct, settings.rag_top_k)
    summary = summarize(results)
    meta = {"timestamp": stamp, "llm": llm.name, "judge": judge.name, "embed": settings.embed_model,
            "n_questions": len(questions), "k": settings.rag_top_k,
            "priced": bool(settings.price_input_per_m or settings.price_output_per_m),
            "wall_time_s": round(time.perf_counter() - t0, 1)}
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary, "ablation": ablation},
                                                 indent=2, ensure_ascii=False))
    (out / "report.md").write_text(report_md(summary, ablation, meta))
    return out
