"""问数 ReAct 子图：agent →（sql / 打回 / 追问）→ agent，step≤30。

取数走官方 tool_calls。结论仍写在正文的 ```conclusion 里。
"""
from __future__ import annotations
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .. import llm
from ..config import MAX_STEPS, PLAN_SOFT, PLAN_STEPS, RESULT_ROWS
from ..schemas import Evidence
from ..tools import sql_tool
from ..tools.batch import classify, formula_of, plan_hit, run_readonly, schema_hit
from ..tools.schemas import assistant_message, calls_to_pending, query_tools, tool_message, worker_tools
from .runtime import GraphContext, current_cursor
from .textutil import _extract_block, _model_line, _rows_preview


QUERY_SYS = """你是问数执行 Agent，根据目标查询标准指标。
【工作方式】
1. 需要取数时调用 run_sql。一次一条 DuckDB 只读 SELECT，表用英文别名，列名用中文原名。不要在正文里写 SQL。
2. 目标含多个时间期（如2025与2024、各季度）、多个指标或多个维度取值时，用一条 SQL 一次取回（按年/季 GROUP BY、或 CASE 对比），不要为每期/每值各跑一次。
3. 拿到真实结果后，在正文输出 ```conclusion：给出具体数值答案，注明口径与时间窗口；多期/多值要分别列出并给出对比。
【硬约束】
- 未成功执行过任何 SQL、没有真实结果前，禁止输出 ```conclusion，禁止只复述口径或给"见查询结果"这类空话。
- 口径以工具返回为准，禁止编造数字。
- 只回答本查询点的数值事实；不要评价"数据不足/无法定位原因"，不要建议下钻、渠道拆分或进一步分析，归因由主 Agent 另行调度。
- 关联和过滤以工具返回的关系与表结构为准。"""

def worker_sys(ontology_enabled: bool) -> str:
    head = (
        "你是一位被派遣的分析专家子 Agent，严格按给定 Skill 的操作指引取数分析。\n"
        "分析目标是全体专家共用的问题，只说明指标、时间和要解释的变化，不是你的任务清单。计划只写你的技能方法里、并且这道题需要的取数。技能里没有的角度不要写进计划，也不要写进结论。\n"
        "先制定计划，再在执行中调用工具、观察结果，并可以修订尚未执行的步骤。\n"
        "用接口提供的函数调用工具，不要在正文里写工具名、参数或 JSON。\n"
        "一次回复最多调用 4 个函数。超过 4 个时，多出来的这一轮不会执行，要在下一次回复里继续。\n"
        "写 SQL、以及按结果修订计划，不要和给它们提供输入的工具放在同一次回复里。\n"
        "修订计划时单独输出 ```plan，只写尚未执行的步骤。可以新增、改写或删除这些步骤，也可以写 {\"steps\":[]} 放弃剩余步骤。计划控制在 "
        + str(PLAN_SOFT)
        + " 步以内，最多 "
        + str(PLAN_STEPS)
        + " 步，已完成的步骤不要再写进去。超过 "
        + str(PLAN_STEPS)
        + " 步的整份计划不会收下。做完 "
        + str(PLAN_SOFT)
        + " 步就建议收拢；已经写在计划里的可以做完，也可以再追加，总数到 "
        + str(PLAN_STEPS)
        + " 步为止。第 "
        + str(PLAN_STEPS)
        + " 步完成后只写 ```conclusion，有结果支撑的判断写上，没有结果的点不写。当前计划的步骤都已有查询结果、总数还没到 "
        + str(PLAN_STEPS)
        + " 时，不要直接查询，先用 ```plan 追加步骤，下一回复再查。同一次回复里，查询不能多于还没执行的步骤。数已经够回答问题时，也可以不必做完每一步，直接写 ```conclusion。\n"
        "计划步骤只写要查的数据，不要把 resolve_certified_metric、read_sql_plan、describe_schema、read_entity 写成步骤。这些工具不占用计划勾选。\n"
        "计划只写这个问题要回答的取数。技能里的其余下钻不是必做清单，和本题无关的不要写进计划。指标用问题点名的那个，技能里的销售额、毛利额只是例子。\n"
        "问题要求的数已经取到后就写 ```conclusion，不必做完技能里的每一条。先用数据给出支撑判断，末尾以「## 该专家的主要结论」作为小标题。结论写判断和依据。多项数字对照时，可以用 Markdown 表格，也可以写成句子，由你判断哪种更好读，不必每次都画表。不要写「分析已完成」「没有剩余步骤」这类执行状态。\n"
        "未成功执行过任何 SQL 之前，禁止输出 ```conclusion。不要调用 list_skills、dispatch、report、finish。\n"
    )
    if ontology_enabled:
        body = (
            "本体访问开启。可以调用 read_entity、read_value_domain、read_relations、read_policies、"
            "read_logical_dataset、resolve_certified_metric、read_sql_plan、describe_schema、run_sql。\n"
            "口径和关联以工具返回为准。写 SQL 之前，更早的回复里必须已经成功调用 resolve_certified_metric 和 read_sql_plan。\n"
            "这两次成功之后就够了。后面的计划步骤直接 run_sql，不要为每一步重新读本体。修订计划不能代替取数。\n"
        )
    else:
        body = (
            "本体访问已关闭。不能调用实体、值域、关系、策略、逻辑数据集、认证指标和 SQL 规划。\n"
            "没有本体和认证指标，先调用 describe_schema 看物理表结构，再写 SQL。describe_schema 成功一次即可，后面的计划步骤直接 run_sql。修订计划不能代替取数。\n"
        )
    return head + body


WORKER_SYS = worker_sys(True)


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
    step_action: str
    pending_calls: list
    metric_ready: bool
    plan_ready: bool
    schema_ready: bool
    metric_formula: str


def _empty_nudge(profile: str) -> str:
    if profile == "query":
        return ("你还没有成功取到任何数据。禁止只给口径或空结论；"
                "请先调用 run_sql 取数，拿到真实结果后再在正文输出 ```conclusion。")
    return ("你还没有成功取到任何数据。禁止只给口径或空泛判断；"
            "请先调用 run_sql 取数，拿到真实结果后再在正文输出 ```conclusion。")


def _ask_nudge(profile: str) -> str:
    if profile == "query":
        return "请调用 run_sql，或在拿到结果后于正文输出 ```conclusion。不要在正文里写 SQL。"
    return "请调用函数取数，或在完成后于正文输出 ```conclusion。不要在正文里写工具参数。"


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

    offered = query_tools() if profile == "query" else worker_tools(True)
    resp = await llm.chat_stream(
        state["messages"], model=state["model"], on_reasoning=_reasoning, tools=offered,
    )
    usage = dict(state["usage"])
    for k in ("prompt", "completion", "total"):
        usage[k] += resp.get("usage", {}).get(k, 0)
    content = resp.get("content", "") or ""
    tool_calls = list(resp.get("tool_calls") or [])
    messages = list(state["messages"]) + [assistant_message(content, resp.get("reasoning") or "", tool_calls)]
    pending = calls_to_pending(tool_calls)
    concl = "" if pending else _extract_block(content, "conclusion")
    action = _model_line(content) if content.strip() else None

    if concl:
        show, kind = concl, "conclusion"
        action = _model_line(concl)
    elif content.strip():
        show, kind = content.strip()[:500], "think"
    else:
        show, kind = "", "think"
    if show:
        await emit({
            "type": "step_log", "skill_id": skill_id, "step": step,
            "kind": kind, "text": show, "action": action,
        })
        # 问数循环里的正文同步进主时间线；工具参数留在 tool_calls，不写进这段文字。
        if profile == "query" and kind == "think" and show.strip():
            await emit({"type": "reasoning", "skill_id": "query", "text": show.strip()})

    sql = next((str(item.get("sql") or "") for item in pending if item.get("tool") == "run_sql"), "")
    update: dict = {
        "messages": messages,
        "usage": usage,
        "step": step,
        "sql": sql,
        "step_action": action or "",
        "pending_calls": pending,
    }
    if pending:
        update["route"] = "sql"
    elif concl and state["evidences"]:
        update["route"] = "done"
        update["conclusion"] = concl
        update["status"] = "done"
    elif concl:
        update["route"] = "reject"
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
    pending = list(state.get("pending_calls") or [])
    if not pending and state.get("sql"):
        pending = [{"tool": "run_sql", "sql": state["sql"], "call_id": "call_sql"}]
    if state["profile"] == "query":
        return await _query_sql(state, emit, pending)
    return await _worker_sql(state, emit, pending)


async def _query_sql(state: ReactState, emit, pending: list[dict]) -> dict:
    messages = list(state["messages"])
    evidences = list(state["evidences"])
    for call in pending:
        call_id = str(call.get("call_id") or "")
        if call.get("tool") != "run_sql":
            messages.append(tool_message(call_id, "未执行：问数执行只调用 run_sql"))
            continue
        res = sql_tool.run_sql(str(call.get("sql") or ""), cursor=current_cursor())
        if res.get("error"):
            messages.append(tool_message(call_id, f"SQL 报错：{res['error']}，修正后重试。"))
            await emit({
                "type": "tool_call", "skill_id": state["skill_id"], "tool": "run_sql",
                "title": "run_sql", "detail": res["error"], "sql": str(call.get("sql") or ""),
            })
            continue
        shown = res["rows"][:RESULT_ROWS]
        ev = Evidence(metric=state["goal"], sql=res["sql"], row_count=res["row_count"], rows=shown)
        evidences.append(ev.to_dict())
        await emit({
            "type": "tool_call", "skill_id": state["skill_id"], "tool": "run_sql",
            "title": (state.get("step_action") or "").strip() or "run_sql",
            "detail": f"返回 {res['row_count']} 行", "sql": res["sql"],
        })
        preview = _rows_preview(shown, res["row_count"])
        messages.append(tool_message(
            call_id, f"结果（{res['row_count']}行）：\n{preview}\n给 ```conclusion。",
        ))
    return {"messages": messages, "evidences": evidences, "pending_calls": []}


async def _worker_sql(state: ReactState, emit, pending: list[dict]) -> dict:
    decisions = classify(
        pending, ontology_enabled=True, metric_ready=bool(state.get("metric_ready")),
        plan_ready=bool(state.get("plan_ready")), schema_ready=bool(state.get("schema_ready")),
        skills_listed=False, has_sql=bool(state.get("evidences")), dispatched=False,
        allow_orchestrator=False,
    )
    messages = list(state["messages"])
    evidences = list(state["evidences"])
    formula = str(state.get("metric_formula") or "")
    metric_ready = bool(state.get("metric_ready"))
    plan_ready = bool(state.get("plan_ready"))
    schema_ready = bool(state.get("schema_ready"))
    for item in decisions:
        tool = item["tool"]
        name = str(tool.get("tool") or "")
        call_id = str(tool.get("call_id") or "")
        if not item["run"]:
            messages.append(tool_message(call_id, "未执行：" + item["reason"]))
            continue
        if name == "run_sql":
            res = sql_tool.run_sql(str(tool.get("sql") or ""), cursor=current_cursor())
            if res.get("error"):
                messages.append(tool_message(call_id, f"SQL 执行报错：{res['error']}。请修正后重试，不要编造结果。"))
                continue
            shown = res["rows"][:RESULT_ROWS]
            ev = Evidence(metric=state["goal"], formula=formula, sql=res["sql"],
                          row_count=res["row_count"], rows=shown)
            evidences.append(ev.to_dict())
            preview = _rows_preview(shown, res["row_count"])
            messages.append(tool_message(
                call_id, f"SQL 结果（{res['row_count']} 行）：\n{preview}\n继续下一步，或给出 ```conclusion。",
            ))
            await emit({
                "type": "tool_call", "skill_id": state["skill_id"], "tool": "run_sql",
                "title": "run_sql", "detail": f"返回 {res['row_count']} 行", "sql": res["sql"],
            })
            continue
        body = run_readonly(tool)
        messages.append(tool_message(call_id, body))
        if name == "resolve_certified_metric":
            got = formula_of(body)
            if got:
                metric_ready = True
                formula = got
        elif name == "read_sql_plan" and plan_hit(body):
            plan_ready = True
        elif name == "describe_schema" and schema_hit(body):
            schema_ready = True
    return {
        "messages": messages,
        "evidences": evidences,
        "pending_calls": [],
        "metric_ready": metric_ready,
        "plan_ready": plan_ready,
        "schema_ready": schema_ready,
        "metric_formula": formula,
    }


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

# 30 步里每步最多 agent + sql 两个超步，再留余量，避免先撞上图的递归上限。
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
            "step_action": "",
            "pending_calls": [],
            "metric_ready": False,
            "plan_ready": False,
            "schema_ready": False,
            "metric_formula": "",
        },
        context=context,
        config={"recursion_limit": REACT_RECURSION_LIMIT},
    )
