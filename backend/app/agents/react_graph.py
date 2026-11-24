"""问数与专家共用的 ReAct 子图：agent →（sql / 打回 / 追问）→ agent，step≤60。

提示词正文从原 supervisor.QUERY_SYS / worker.WORKER_SYS 原样迁入，供本图节点使用。
"""
from __future__ import annotations
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .. import llm
from ..config import MAX_STEPS
from ..schemas import Evidence
from ..tools import sql_tool
from .runtime import GraphContext, current_cursor
from .textutil import _action_before_sql, _extract_block, _extract_sql, _rows_preview


# 提示词正文不改，只是随 ReAct 子图从 supervisor / worker 挪到这里。
QUERY_SYS = """你是问数执行 Agent，根据目标查询标准指标。
【工作方式】
1. 先看下方"数据表与字段"确定表别名、列名与允许 join，禁止用 information_schema 盲探表结构。
2. 需要取数时输出一条 ```sql（DuckDB 只读，表用英文别名，列名用中文原名）。
3. 目标含多个时间期（如2025与2024、各季度）、多个指标或多个维度取值时，用一条 SQL 一次取回（按年/季 GROUP BY、或 CASE 对比），不要为每期/每值各跑一次。
4. 拿到真实结果后再输出 ```conclusion：给出具体数值答案，注明口径与时间窗口；多期/多值要分别列出并给出对比。
【硬约束】
- 未成功执行过任何 SQL、没有真实结果前，禁止输出 ```conclusion，禁止只复述口径或给"见查询结果"这类空话。
- 只用本体标准指标口径，禁止编造数字；金额=SUM(实付金额)-SUM(退款金额)，订单量=COUNT(DISTINCT 订单ID)。
- 只回答本查询点的数值事实；不要评价"数据不足/无法定位原因"，不要建议下钻、渠道拆分或进一步分析，归因由主 Agent 另行调度。
- 时间过滤：orders.订单日期 可直接按年/季过滤；需要促销标记/标准年月时再 join d_date。
表别名与数据地图：{data_map}"""

WORKER_SYS = """你是一位被派遣的分析专家子 Agent，严格按给定 Skill 的操作指引取数分析。
【工作方式】
1. 先看下方"数据表与字段"确定表别名、列名与允许 join，禁止用 information_schema 盲探表结构。
2. 每一步只做一件事：需要取数时输出一条 DuckDB 只读 SQL（```sql 代码块），表用英文别名，列用中文原名；输出 SQL 前先用一句完整、明确的中文写明本步动作，格式"步骤N：<明确的查询对象 + 口径 + 时间/范围>"，例如"步骤一：分别统计2024与2025全年销售额并计算同比变化"。必须写清查什么、按什么口径、看哪个范围；禁止半句、禁止未闭合括号的残句、禁止只写"订单量）"这类碎片。
3. 拿到 SQL 结果后继续下一步；数据不足可修正 SQL 重试（在 step 上限内）。
4. 全部取数完成后，输出最终结论 ```conclusion 代码块：先用数据给出支撑判断，末尾以"## 该专家的主要结论"作为小标题，给出本视角下的一句话核心结论。
【硬约束】
- 未成功执行过任何 SQL、没有真实结果前，禁止输出 ```conclusion，禁止只复述口径或空泛判断。
- 只用下方本体给定的标准指标口径，禁止自造指标或编造数字，结论必须基于真实 SQL 结果。
- 时间过滤：orders.订单日期 可直接按年/季过滤；需要促销标记/标准年月时再 join d_date。
数据表与数据地图：{data_map}"""


class ReactState(TypedDict):
    messages: list[dict]
    model: str
    skill_id: str
    skill_name: str
    goal: str
    profile: str          # query | worker
    step: int
    status: str
    conclusion: str
    evidences: list[dict]
    usage: dict
    route: str
    sql: str


def _empty_nudge(profile: str) -> str:
    if profile == "query":
        return ("你还没有成功取到任何数据。禁止只给口径或空结论；"
                "请先输出 ```sql 取数，拿到真实结果后再 ```conclusion。")
    return ("你还没有成功取到任何数据。禁止只给口径或空泛判断；"
            "请先输出 ```sql 取数，拿到真实结果后再 ```conclusion。")


def _ask_nudge(profile: str) -> str:
    if profile == "query":
        return "请输出 ```sql 或 ```conclusion。"
    return "请输出 ```sql 取数，或在完成后输出 ```conclusion。"


def _truncated_text(profile: str) -> str:
    if profile == "query":
        return "达到 step 上限，未完成（不伪造）。"
    return "达到 step 上限，结论不完整（已中止，不伪造完成）。"


async def agent_node(state: ReactState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    step = state["step"] + 1
    skill_id = state["skill_id"]
    profile = state["profile"]
    await emit({
        "type": "agent_progress", "skill_id": skill_id,
        "skill_name": state["skill_name"], "step": step,
    })

    async def _reasoning(t: str):
        await emit({"type": "reasoning", "skill_id": skill_id, "text": t})

    resp = await llm.chat_stream(state["messages"], model=state["model"], on_reasoning=_reasoning)
    usage = dict(state["usage"])
    for k in ("prompt", "completion", "total"):
        usage[k] += resp.get("usage", {}).get(k, 0)
    content = resp.get("content", "")
    messages = list(state["messages"]) + [{"role": "assistant", "content": content}]
    sql = _extract_sql(content)
    concl = _extract_block(content, "conclusion")

    if sql:
        show, kind = sql, "sql"
        action = _action_before_sql(content) if profile == "query" else _action_before_sql(content, sql)
    elif concl:
        show, kind = concl, "conclusion"
        action = "形成查询结论" if profile == "query" else "形成分析结论"
    else:
        show, kind = content.strip()[:500], "think"
        action = None
    if show:
        await emit({
            "type": "step_log", "skill_id": skill_id, "step": step,
            "kind": kind, "text": show, "action": action,
        })
        # 问数循环里的自语思路同步进主时间线；专家思路只留在 step_log
        if profile == "query" and kind == "think" and show.strip():
            await emit({"type": "reasoning", "skill_id": "query", "text": show.strip()})

    update: dict = {
        "messages": messages,
        "usage": usage,
        "step": step,
        "sql": sql or "",
    }
    if concl and state["evidences"]:
        update["route"] = "done"
        update["conclusion"] = concl
        update["status"] = "done"
    elif concl:
        update["route"] = "reject"
    elif sql:
        update["route"] = "sql"
    else:
        update["route"] = "ask"
    return update


def after_agent(state: ReactState) -> str:
    route = state.get("route") or "ask"
    if route == "done":
        return END
    # 最后一步若给出了 SQL，仍要执行（与原 for 循环一致），执行完再截断，不再调模型。
    if route == "sql":
        return "sql"
    if state["step"] >= MAX_STEPS:
        return "truncate"
    if route == "reject":
        return "reject"
    return "ask"


async def sql_node(state: ReactState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    res = sql_tool.run_sql(state["sql"], cursor=current_cursor())
    messages = list(state["messages"])
    if res.get("error"):
        if state["profile"] == "query":
            messages.append({"role": "user", "content": f"SQL 报错：{res['error']}，修正后重试。"})
        else:
            messages.append({"role": "user",
                             "content": f"SQL 执行报错：{res['error']}。请修正后重试，不要编造结果。"})
        return {"messages": messages}
    ev = Evidence(metric=state["goal"], sql=res["sql"], row_count=res["row_count"],
                  rows=res["rows"][:50])
    evidences = list(state["evidences"]) + [ev.to_dict()]
    await emit({
        "type": "tool_call", "skill_id": state["skill_id"], "tool": "run_sql",
        "title": "执行查询 · run_sql",
        "detail": f"返回 {res['row_count']} 行", "sql": res["sql"],
    })
    preview = _rows_preview(res["rows"][:10])
    if state["profile"] == "query":
        messages.append({"role": "user",
                         "content": f"结果（{res['row_count']}行）：\n{preview}\n给 ```conclusion。"})
    else:
        messages.append({"role": "user",
                         "content": f"SQL 结果（{res['row_count']} 行）：\n{preview}\n继续下一步，或给出 ```conclusion。"})
    return {"messages": messages, "evidences": evidences}


async def reject_node(state: ReactState, runtime: Runtime) -> dict:
    del runtime
    messages = list(state["messages"])
    messages.append({"role": "user", "content": _empty_nudge(state["profile"])})
    return {"messages": messages}


async def ask_node(state: ReactState, runtime: Runtime) -> dict:
    del runtime
    messages = list(state["messages"])
    messages.append({"role": "user", "content": _ask_nudge(state["profile"])})
    return {"messages": messages}


def after_side(state: ReactState) -> str:
    if state["step"] >= MAX_STEPS:
        return "truncate"
    return "agent"


async def truncate_node(state: ReactState, runtime: Runtime) -> dict:
    del runtime
    return {
        "status": "truncated",
        "conclusion": state["conclusion"] or _truncated_text(state["profile"]),
    }


def _build_react():
    g = StateGraph(ReactState, context_schema=GraphContext)
    g.add_node("agent", agent_node)
    g.add_node("sql", sql_node)
    g.add_node("reject", reject_node)
    g.add_node("ask", ask_node)
    g.add_node("truncate", truncate_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["sql", "reject", "ask", "truncate", END])
    for name in ("sql", "reject", "ask"):
        g.add_conditional_edges(name, after_side, ["agent", "truncate"])
    g.add_edge("truncate", END)
    return g.compile(name="qa-react")


REACT = _build_react()

# 60 步里每步最多 agent + sql 两个超步，再留余量，避免先撞上图的递归上限。
REACT_RECURSION_LIMIT = MAX_STEPS * 4 + 16


async def run_react(
    *,
    messages: list[dict],
    model: str,
    skill_id: str,
    skill_name: str,
    goal: str,
    profile: str,
    context: GraphContext,
) -> dict:
    return await REACT.ainvoke(
        {
            "messages": messages,
            "model": model,
            "skill_id": skill_id,
            "skill_name": skill_name,
            "goal": goal,
            "profile": profile,
            "step": 0,
            "status": "running",
            "conclusion": "",
            "evidences": [],
            "usage": {"prompt": 0, "completion": 0, "total": 0},
            "route": "",
            "sql": "",
        },
        context=context,
        config={"recursion_limit": REACT_RECURSION_LIMIT},
    )
