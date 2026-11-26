"""一次模型回复里的工具批处理。超过 4 个只处理前 4 个，其余退回给模型。"""
from __future__ import annotations
import json

from . import knowledge_tools, skill_tools, sql_tool

MAX_TOOLS = 4

ONTOLOGY_TOOLS = {"read_entity", "read_value_domain", "read_relations", "read_policies"}
SEMANTIC_TOOLS = {"read_logical_dataset", "resolve_certified_metric", "read_sql_plan"}
KNOWLEDGE_TOOLS = ONTOLOGY_TOOLS | SEMANTIC_TOOLS
ORCH_TOOLS = {"list_skills", "dispatch", "report", "finish"}


_NESTED_KEYS = ("args", "arguments", "input", "parameters")


_TOOL_NAMES = KNOWLEDGE_TOOLS | ORCH_TOOLS | {"run_sql", "describe_schema"}


def normalize_tool(tool: dict) -> dict:
    """兼容 name + arguments 这种写法，并把里层字段提到最外层。"""
    if not isinstance(tool, dict):
        return tool
    if not tool.get("tool") and tool.get("name") in _TOOL_NAMES:
        tool["tool"] = tool.pop("name")
    for key in _NESTED_KEYS:
        inner = tool.get(key)
        if isinstance(inner, str) and inner.strip().startswith("{"):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                inner = None
        if not isinstance(inner, dict):
            continue
        for field, value in inner.items():
            if tool.get(field) in (None, ""):
                tool[field] = value
        tool.pop(key, None)
    return tool


def _arg(tool: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = tool.get(key)
        if value not in (None, ""):
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return default


def label(tool: dict) -> str:
    name = str(tool.get("tool") or "")
    args = {k: v for k, v in tool.items() if k != "tool"}
    return f"{name} {json.dumps(args, ensure_ascii=False)}"


def classify(tools: list[dict], *, ontology_enabled: bool, metric_ready: bool, plan_ready: bool,
             schema_ready: bool, skills_listed: bool, has_sql: bool, dispatched: bool,
             allow_orchestrator: bool, reported: bool = False) -> list[dict]:
    """返回 [{tool, run, reason}]。run 为假时 reason 写给模型。"""
    head = [normalize_tool(t) for t in tools[:MAX_TOOLS]]
    tail = [normalize_tool(t) for t in tools[MAX_TOOLS:]]
    names = {str(t.get("tool") or "") for t in head}
    total = len(tools)
    out = []
    for tool in head:
        reason = _reason(
            tool, names=names, total=total, ontology_enabled=ontology_enabled,
            metric_ready=metric_ready, plan_ready=plan_ready, schema_ready=schema_ready,
            skills_listed=skills_listed, has_sql=has_sql, dispatched=dispatched,
            allow_orchestrator=allow_orchestrator, reported=reported,
        )
        out.append({"tool": tool, "run": reason is None, "reason": reason or ""})
    for tool in tail:
        out.append({"tool": tool, "run": False, "reason": "单轮最多 4 个工具"})
    return out


def _reason(tool: dict, *, names: set[str], total: int, ontology_enabled: bool,
            metric_ready: bool, plan_ready: bool, schema_ready: bool, skills_listed: bool,
            has_sql: bool, dispatched: bool, allow_orchestrator: bool, reported: bool) -> str | None:
    name = str(tool.get("tool") or "")
    if name in ORCH_TOOLS and not allow_orchestrator:
        return "子 Agent 不能使用这个工具"
    if name in KNOWLEDGE_TOOLS and not ontology_enabled:
        return "本体访问已关闭"
    if name == "run_sql":
        providers = {"resolve_certified_metric", "read_sql_plan"} if ontology_enabled else {"describe_schema"}
        if providers & names:
            return "同批里有尚未看见结果的查询，SQL 这一轮不执行"
        if ontology_enabled and not (metric_ready and plan_ready):
            return "还没有成功调用 resolve_certified_metric 和 read_sql_plan"
        if not ontology_enabled and not schema_ready:
            return "还没有成功调用 describe_schema"
        if not str(tool.get("sql") or "").strip():
            keys = "、".join(k for k in tool if k != "tool") or "（没有其他字段）"
            return f"run_sql 缺少 sql。收到的字段：{keys}。sql 要放在最外层。"
        return None
    if name == "dispatch":
        if "run_sql" in names or "report" in names:
            return "同批里还要先把数值报告给用户，这一轮不派工"
        if "list_skills" in names:
            return "同批里有 list_skills，还没看见技能列表，这一轮不派工"
        if not skills_listed:
            return "还没有 list_skills"
        if has_sql and not reported:
            return "数值还没有 report 给用户"
        if dispatched:
            return "这轮已经派过专家"
        return None
    if name == "report":
        if "run_sql" in names:
            return "同批里有尚未返回的 SQL，这一轮不报告"
        if not has_sql:
            return "还没有成功的 SQL 结果"
        if not str(tool.get("text") or "").strip():
            return "report 缺少 text"
        return None
    if name == "list_skills":
        if "run_sql" in names:
            return "同批里有尚未返回的 SQL，这一轮不查看专家"
        if "report" in names:
            return "同批里有 report，先把数值报告给用户，这一轮不查看专家"
        return None
    if name == "finish":
        if total > 1:
            return "同批还有别的工具，这一轮不收口"
        return None
    if name in KNOWLEDGE_TOOLS or name == "describe_schema":
        return None
    if not name:
        return "缺少 tool"
    return f"未知工具 {name}"


def run_readonly(tool: dict, session_id: str | None = None) -> str:
    name = str(tool.get("tool") or "")
    if name == "read_entity":
        return knowledge_tools.read_entity(_arg(tool, "name", "entity", "entity_name"))
    if name == "read_value_domain":
        return knowledge_tools.read_value_domain(_arg(tool, "name", "domain"))
    if name == "read_relations":
        return knowledge_tools.read_relations(_arg(tool, "entity", "name"))
    if name == "read_policies":
        return knowledge_tools.read_policies(_arg(tool, "topic", "id", "name"))
    if name == "read_logical_dataset":
        return knowledge_tools.read_logical_dataset(_arg(tool, "name", "dataset"))
    if name == "resolve_certified_metric":
        return knowledge_tools.resolve_certified_metric(_arg(tool, "metric", "name", "metric_name", "query", "question"))
    if name == "read_sql_plan":
        return knowledge_tools.read_sql_plan(_arg(tool, "metric", "name", "metric_name", "query", "question"))
    if name == "describe_schema":
        return sql_tool.describe_schema(_arg(tool, "table", "name"))
    if name == "list_skills":
        return skill_tools.list_skills(session_id)
    return f"未知工具 {name}"


def plan_hit(body: str) -> bool:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False
    plans = data.get("plans") if isinstance(data, dict) else None
    return isinstance(plans, list) and bool(plans)


def schema_hit(body: str) -> bool:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False
    if isinstance(data, dict) and data.get("error"):
        return False
    return isinstance(data, list) and any(isinstance(item, dict) and item.get("columns") for item in data)


def formula_of(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if isinstance(data, dict):
        return str(data.get("口径") or "")
    return ""
