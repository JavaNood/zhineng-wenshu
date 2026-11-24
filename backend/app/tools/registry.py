"""工具注册表：登记工具说明与分发（function calling 风格的工具目录）。"""
from __future__ import annotations
from typing import Any, Callable

from . import ontology_tools, sql_tool, skill_tools

# 给 LLM 的工具说明（主 Agent 可调用的只读目录工具；run_sql 由执行循环内联拦截，不在此注册）
TOOL_SPECS = [
    {"name": "read_ontology", "desc": "读取指标语义层：表/字段/指标口径/维度/关系。可选 topic 限定。"},
    {"name": "resolve_metric", "desc": "把自然语言指标对齐本体标准指标与口径，禁止自造口径。"},
    {"name": "list_skills", "desc": "列出技能库可用分析 Skill（专家）。"},
    {"name": "list_dimensions", "desc": "列出可下钻维度与枚举值，避免猜枚举。"},
]


def list_dimensions_text() -> str:
    from ..ontology import loader as onto
    import json
    return json.dumps(
        [{"code": d["code"], "name": d["name"], "values": d.get("values")}
         for d in onto.list_dimensions()],
        ensure_ascii=False, indent=2)


# 主 Agent 可用工具（子 Agent 子集在 worker 内单独开放）
HANDLERS: dict[str, Callable[..., str]] = {
    "read_ontology": lambda **a: ontology_tools.read_ontology(a.get("topic", "")),
    "resolve_metric": lambda **a: ontology_tools.resolve_metric(a.get("metric", "")),
    "list_skills": lambda **a: skill_tools.list_skills(),
    "list_dimensions": lambda **a: list_dimensions_text(),
}


def execute_tool(name: str, args: dict[str, Any] | None = None) -> str:
    handler = HANDLERS.get(name)
    if not handler:
        return f"未知工具 {name}，可用：{list(HANDLERS)}"
    return handler(**(args or {}))
