from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class Chunk:
    page: int  # 1-based PDF page index (as shown in a PDF viewer)
    kind: str  # "text" or "table"
    text: str


@dataclass
class Hit:
    chunk_id: int
    bank: str
    year: int
    page: int
    kind: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "bank": self.bank,
            "year": self.year,
            "page": self.page,
            "kind": self.kind,
            "text": self.text,
            "score": round(self.score, 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hit":
        return cls(**{k: d[k] for k in ("chunk_id", "bank", "year", "page", "kind", "text")}, score=d.get("score", 0.0))


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.llm_calls += other.llm_calls


@dataclass
class Answer:
    question: str
    mode: str
    answer: str
    hits: list[Hit]  # everything retrieved (used for recall@k)
    context: list[Hit]  # what the LLM actually saw, in [S#] order
    cited: list[int]  # S-numbers cited in the answer
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    trace: list[str] = field(default_factory=list)

    def citations(self) -> list[dict]:
        out = []
        for n in self.cited:
            if 1 <= n <= len(self.context):
                h = self.context[n - 1]
                out.append({"source": f"S{n}", "bank": h.bank, "year": h.year, "page": h.page})
        return out


# ---- structured LLM outputs -------------------------------------------------

class SubQuery(BaseModel):
    query: str = Field(description="A self-contained search query, phrased like text in an annual report.")
    bank: str | None = Field(default=None, description="Bank code to filter on, or null to search all banks.")
    year: int | None = Field(default=None, description="Fiscal year to filter on, or null.")


class Plan(BaseModel):
    sub_queries: list[SubQuery] = Field(description="1 to 6 sub-queries. One per bank and metric for comparisons.")


class Check(BaseModel):
    grounded: bool = Field(description="True if every factual claim in the answer is supported by the cited sources.")
    complete: bool = Field(description="True if the answer addresses every part of the question.")
    missing: str = Field(default="", description="What is missing or unsupported, phrased as what to search for next.")


class Judgement(BaseModel):
    correct: bool = Field(description="True if the candidate answer matches the reference on every requested fact.")
    reason: str = Field(description="One sentence.")
