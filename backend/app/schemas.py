"""内部数据结构：意图拆分、Plan、证据、子结论、汇聚结果。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .config import RESULT_ROWS


@dataclass
class PlanStep:
    step_id: str
    action: str
    depends_on: list[str] = field(default_factory=list)
    query_spec: str = ""
    output: str = ""


@dataclass
class Plan:
    intent: str            # chat / query / analysis
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)  # 分析意图选中的 Skill


@dataclass
class IntentPoint:
    index: int
    raw_text: str
    intent: str            # chat / query / analysis
    plan: Plan | None = None


@dataclass
class Evidence:
    """可回溯证据：指标定义 / SQL / 结果快照 / 口径公式。"""
    metric: str = ""
    formula: str = ""
    sql: str = ""
    window: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "formula": self.formula,
            "sql": self.sql,
            "window": self.window,
            "row_count": self.row_count,
            "rows": self.rows[:RESULT_ROWS],
        }


@dataclass
class SubResult:
    skill_id: str
    skill_name: str
    conclusion: str
    evidences: list[Evidence] = field(default_factory=list)
    status: str = "done"          # done / error / truncated
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "conclusion": self.conclusion,
            "status": self.status,
            "note": self.note,
            "evidences": [e.to_dict() for e in self.evidences],
        }


@dataclass
class FinalResult:
    summary: str = ""
    consensus: list[dict] = field(default_factory=list)
    divergences: list[dict] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "consensus": self.consensus,
            "divergences": self.divergences,
            "evidences": [e.to_dict() for e in self.evidences],
            "tokens": self.tokens,
        }
