"""官方 function calling 的工具定义，以及 tool_calls 和内部工具字典的转换。"""
from __future__ import annotations
import json
import uuid

from .batch import normalize_tool


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _str(description: str) -> dict:
    return {"type": "string", "description": description}


_KNOWLEDGE = [
    _fn("read_entity", "按实体名读取属性和标识。", {"name": _str("实体名")}, ["name"]),
    _fn("read_value_domain", "按值域名读取枚举或取值来源。", {"name": _str("值域名")}, ["name"]),
    _fn("read_relations", "按实体读取允许的关系。", {"entity": _str("实体名")}, ["entity"]),
    _fn("read_policies", "按主题读取策略正文。", {"topic": _str("策略主题")}, ["topic"]),
    _fn("read_logical_dataset", "读取逻辑数据集的列、类型和粒度。", {"name": _str("逻辑数据集或视图名")}, ["name"]),
    _fn(
        "resolve_certified_metric",
        "把自然语言指标对齐认证指标，返回口径。",
        {"metric": _str("用户点名的指标名")},
        ["metric"],
    ),
    _fn(
        "read_sql_plan",
        "读取指标的 SQL 规划约束。不返回可执行 SQL。",
        {"metric": _str("用户点名的指标名")},
        ["metric"],
    ),
]

_SCHEMA = _fn(
    "describe_schema",
    "读取视图别名、来源文件、列名和类型。",
    {"table": _str("视图名，可空")},
    [],
)

_SQL = _fn(
    "run_sql",
    "执行一条 DuckDB 只读 SELECT。只查用户点名的指标、时间和维度。",
    {"sql": _str("一条 SELECT。列用中文原名。")},
    ["sql"],
)

_REPORT = _fn(
    "report",
    "把已经查到的数值告诉用户。只写用户点名的指标、时间和维度，不要附带没问过的拆分。",
    {"text": _str("给用户看的数值答案，数字只能来自工具返回")},
    ["text"],
)

_LIST = _fn("list_skills", "列出分析专家。", {}, [])

_DISPATCH = _fn(
    "dispatch",
    "派专家解释原因。一轮最多 3 位。没有本会话融合专家时至少 2 位；选了融合专家时可以只派 1 位。"
    "不要把融合专家和他的父专家或更上层专家派在同一轮。有 SQL 结果时必须先 report。",
    {
        "skills": {"type": "array", "items": {"type": "string"}, "description": "skill_id 列表"},
        "goal": _str("这一次要专家回答的问题"),
    },
    ["skills", "goal"],
)

_FINISH = _fn(
    "finish",
    "结束这轮对话。summary 使用已经报告的数值或专家结论。",
    {
        "summary": _str("整体结论"),
        "consensus": {
            "type": "array",
            "description": "至少两个专家共同指向的判断",
            "items": {
                "type": "object",
                "properties": {
                    "point": _str("判断"),
                    "support": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "divergences": {
            "type": "array",
            "description": "专家结论互相冲突的地方",
            "items": {
                "type": "object",
                "properties": {
                    "point": _str("分歧"),
                    "why": _str("原因"),
                    "pending": _str("待复核"),
                },
            },
        },
    },
    ["summary"],
)


def main_tools(ontology_enabled: bool) -> list[dict]:
    head = _KNOWLEDGE if ontology_enabled else []
    return head + [_SCHEMA, _SQL, _REPORT, _LIST, _DISPATCH, _FINISH]


def worker_tools(ontology_enabled: bool) -> list[dict]:
    head = _KNOWLEDGE if ontology_enabled else []
    return head + [_SCHEMA, _SQL]


def query_tools() -> list[dict]:
    return [_SQL]


def assistant_message(content: str, reasoning: str, tool_calls: list[dict] | None) -> dict:
    """带 tools 的后续请求必须带回 reasoning_content。"""
    msg = {
        "role": "assistant",
        "content": content or "",
        "reasoning_content": reasoning or "",
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _loads(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def calls_to_pending(tool_calls: list[dict] | None) -> list[dict]:
    """把官方 tool_calls 收成批处理用的工具字典，并保留 call_id。"""
    pending = []
    for index, call in enumerate(tool_calls or []):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        args = _loads(fn.get("arguments"))
        call_id = str(call.get("id") or f"call_{index}_{uuid.uuid4().hex[:8]}")
        tool = normalize_tool({"tool": name, **args})
        if not isinstance(tool, dict):
            tool = {"tool": name}
        tool["tool"] = str(tool.get("tool") or name)
        tool["call_id"] = call_id
        for key in ("skills", "consensus", "divergences"):
            value = tool.get(key)
            if isinstance(value, str) and value.strip()[:1] in "[{":
                try:
                    tool[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        pending.append(tool)
    return pending
