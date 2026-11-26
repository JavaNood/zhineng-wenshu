"""主 Agent 的 ReAct：一次回复可以调用多个互不依赖的工具，结果合成一次状态更新。"""
from __future__ import annotations
import re

from langgraph.runtime import Runtime

from .. import llm
from ..config import MAX_SUB_AGENTS, RESULT_ROWS
from ..schemas import Evidence, FinalResult
from ..skills import loader as skilllib
from ..tools import sql_tool
from ..tools.batch import classify, formula_of, label, plan_hit, run_readonly, schema_hit
from ..tools.schemas import assistant_message, calls_to_pending, main_tools, tool_message
from .textutil import _rows_preview

MAIN_MAX_TURNS = 32

_WHY = re.compile(r"原因|为什么|为何|下降|诊断|异常")

_BATCH_RULES = """
用接口提供的函数调用工具。不要在正文里写工具名、参数、tool_use 或 JSON。
一次回复最多调用 4 个函数。只调用一个时也用函数调用。超过 4 个时，多出来的这一轮不会执行，需要在看到结果后的下一次回复里继续调用。
互不依赖的查询可以放在同一次回复里。写 SQL、向用户报告、查看专家、派工、收口，都要分成不同的回复：先看见上一步的结果，再做下一步。
正文只用一两句中文，说明这一步要做什么。这两句会直接展示给用户。没有要补充的话时，正文留空。
查询和报告只覆盖用户点名的指标、时间和维度。用户没点名的渠道、订单状态、门店、商品、客群等，不要写进 SQL，也不要写进 report 或 finish。
用户要原因、为什么、变化、诊断时，先只报告点名的数值。解释交给专家：report 成功后再 list_skills、再 dispatch。不要用额外 SQL 自己拆解。
"""

_ON_TOOLS = """
本体访问开启。可以调用 read_entity(name)、read_value_domain(name)、read_relations(entity)、read_policies(topic)、read_logical_dataset(name)、resolve_certified_metric(metric)、read_sql_plan(metric)、describe_schema(table)。
写 SQL 之前，更早的回复里必须已经成功调用过 resolve_certified_metric 和 read_sql_plan。口径、枚举和关联只以工具返回为准。
"""

_OFF_TOOLS = """
本体访问已关闭。不能调用实体、值域、关系、策略、逻辑数据集、认证指标、SQL 规划。调用也会被拒绝。
可以调用 describe_schema(table) 查看物理列名和类型。
写 SQL 之前，更早的回复里必须已经成功调用过 describe_schema。没有认证口径可遵守，按列名自己理解后写 SQL。
"""

_COMMON_TOOLS = """
还可以调用这些函数：
- run_sql(sql)：一条 DuckDB 只读 SELECT。列用中文原名，视图用 describe_schema 或逻辑数据集给出的名字。只查用户点名的指标、时间和维度。
- report(text)：已经有 SQL 结果时，单独一次回复先把数值告诉用户。text 里的数字只能来自工具返回，并且只写用户点名的指标、时间和维度，不要附带没问过的拆分。不要和 list_skills、dispatch、run_sql 放在同一次回复里。
- list_skills()：用户还要解释原因、诊断或对比时，在 report 成功之后的下一次回复单独调用。
- dispatch(skills, goal)：从刚看过的技能里选 2 到 3 个。没调用过 list_skills 不要派。有 SQL 结果但还没 report 时不要派。
  给用户看的那一两句必须使用 list_skills 返回的 name。例如：我派三位专家并行分析：量价结构拆解专家-小王拆解定位订单量和客单价的贡献，时序季节诊断专家-小黄诊断排除季节性因素。
- finish(summary, consensus, divergences)：结束这轮对话。寒暄、与数据无关的问题直接 finish，不要查数。
  只问了数值并且已经 report 过，可以 finish，summary 用已经报告的数值。
  派过专家之后，必须等专家结论返回再 finish：至少两个专家共同指向的写进 consensus，互相冲突的写进 divergences，没有冲突就给空数组。不要编造专家没说过的判断。
  未覆盖、证据不足、某步没取成，不要写进 summary、consensus、divergences。
禁止编造数字和指标口径。
"""


def main_sys(ontology_enabled: bool) -> str:
    head = "你是智能问数的主 Agent。看上一批工具的返回，决定下一步。"
    tools = _ON_TOOLS if ontology_enabled else _OFF_TOOLS
    return head + _BATCH_RULES + tools + _COMMON_TOOLS


MAIN_SYS = main_sys(True)


def _preface(content: str) -> str:
    text = content or ""
    cut = text.find("```")
    if cut >= 0:
        text = text[:cut]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:2])[:300]


def initial_messages(question: str, ontology_enabled: bool = True) -> list[dict]:
    return [
        {"role": "system", "content": main_sys(ontology_enabled)},
        {"role": "user", "content": question},
    ]


def _add_usage(resp_usage: dict, session) -> None:
    session.add_tokens(resp_usage or {})


async def main_turn(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    step = int(state.get("step") or 0) + 1
    if step > MAIN_MAX_TURNS:
        await emit({"type": "error", "message": "主 Agent 达到步数上限，已停止。"})
        return {"step": step, "route": "stop"}

    streamed_act = False

    async def _on_reasoning(text: str) -> None:
        if text:
            await emit({"type": "reasoning", "skill_id": "main", "text": text})

    async def _on_content(text: str) -> None:
        nonlocal streamed_act
        if text:
            streamed_act = True
            await emit({"type": "act", "skill_id": "main", "text": text})

    resp = await llm.chat_stream(
        state["messages"], model=state["model"],
        on_reasoning=_on_reasoning, on_content=_on_content,
        tools=main_tools(bool(state.get("ontology_enabled", True))),
    )
    _add_usage(resp.get("usage") or {}, session)
    content = resp.get("content") or ""
    tool_calls = resp.get("tool_calls") or []
    messages = list(state["messages"]) + [assistant_message(content, resp.get("reasoning") or "", tool_calls)]
    preface = _preface(content)
    if preface and not streamed_act:
        await emit({"type": "act", "skill_id": "main", "text": preface})

    tools = calls_to_pending(tool_calls)
    base = {"messages": messages, "step": step, "pending": tools, "nudge": ""}
    if not tools:
        return {**base, "route": "nudge", "nudge": (
            "没有收到函数调用。请使用接口提供的函数，不要在正文里写工具 JSON。"
            "一次最多 4 个。只修正刚才没调用成功的那一次，保持用户点名的指标、时间和维度，不要另外增加渠道、订单状态或其他拆分。"
        )}
    if len(tools) == 1 and str(tools[0].get("tool") or "") == "finish":
        question = state["question"]
        if _WHY.search(question) and not state.get("dispatched") and not state.get("why_nudged"):
            return {
                **base, "route": "nudge", "why_nudged": True,
                "nudge": "工具没有执行：这个问题在问原因或诊断，还没有派出专家。请先 list_skills，再 dispatch 至少 2 个技能，不要直接 finish。",
            }
        await _emit_finish(state, tools[0], preface, content, emit, session)
        return {**base, "route": "stop"}
    return {**base, "route": "tools"}


async def _emit_finish(state, tool, preface, content, emit, session) -> None:
    summary = str(tool.get("summary") or preface or "").strip() or (content or "").strip()
    consensus = tool.get("consensus") if isinstance(tool.get("consensus"), list) else []
    divergences = tool.get("divergences") if isinstance(tool.get("divergences"), list) else []
    evidences = list(state.get("evidences") or [])
    for pkt in state.get("worker_packets") or []:
        evidences.extend((pkt.get("sub") or {}).get("evidences") or [])
    if not state.get("has_sql") and not state.get("dispatched"):
        await emit({"type": "chat_answer", "text": summary})
        return
    if state.get("dispatched"):
        final = FinalResult(summary=summary, consensus=consensus, divergences=divergences)
        final.evidences = []
        payload = final.to_dict()
        payload["evidences"] = evidences
        session.set_final(payload)
        await emit({"type": "consensus", "consensus": consensus, "divergences": divergences})
        await emit({"type": "final", "summary": summary, "evidences": evidences})
        return
    if not state.get("reported") and summary:
        await emit({"type": "query_answer", "text": summary, "evidences": evidences})


async def nudge_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    note = state.get("nudge") or "请改用函数调用。一次最多 4 个。"
    await emit({
        "type": "tool_call", "skill_id": "main", "tool": "nudge",
        "title": "未执行", "detail": note,
    })
    messages = list(state["messages"])
    last = messages[-1] if messages else {}
    calls = last.get("tool_calls") if isinstance(last, dict) else None
    if calls:
        for call in calls:
            messages.append(tool_message(str(call.get("id") or ""), "未执行：" + note))
    else:
        messages.append({"role": "user", "content": note})
    return {"messages": messages, "route": ""}


async def tools_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    tools = list(state.get("pending") or [])
    decisions = classify(
        tools,
        ontology_enabled=bool(state.get("ontology_enabled", True)),
        metric_ready=bool(state.get("metric_ready")),
        plan_ready=bool(state.get("plan_ready")),
        schema_ready=bool(state.get("schema_ready")),
        skills_listed=bool(state.get("skills_listed")),
        has_sql=bool(state.get("has_sql")),
        dispatched=bool(state.get("dispatched")),
        reported=bool(state.get("reported")),
        allow_orchestrator=True,
    )
    executed: list[tuple[dict, str]] = []
    skipped: list[tuple[dict, str]] = []
    evidences = []
    updates: dict = {"route": ""}
    formula = str(state.get("metric_formula") or "")
    for item in decisions:
        tool = item["tool"]
        name = str(tool.get("tool") or "")
        if not item["run"]:
            skipped.append((tool, item["reason"]))
            await emit({
                "type": "tool_call", "skill_id": "main", "tool": name or "unknown",
                "title": "未执行", "detail": item["reason"], "body": label(tool),
            })
            continue
        if name == "run_sql":
            sql = str(tool.get("sql") or "")
            res = sql_tool.run_sql(sql, cursor=sql_tool.new_cursor())
            if res.get("error"):
                body = res["error"]
                detail = body
            else:
                shown = res["rows"][:RESULT_ROWS]
                body = f"结果（{res['row_count']} 行）：\n{_rows_preview(shown, res['row_count'])}"
                detail = f"返回 {res['row_count']} 行"
                ev = Evidence(metric=state.get("question") or "", formula=formula, sql=res["sql"],
                              row_count=res["row_count"], rows=shown)
                evidences.append(ev.to_dict())
                updates["has_sql"] = True
            executed.append((tool, body))
            await emit({
                "type": "tool_call", "skill_id": "main", "tool": "run_sql",
                "title": "run_sql", "detail": detail, "sql": res.get("sql") or sql,
            })
            continue
        if name == "dispatch":
            picked = [s for s in (tool.get("skills") or []) if isinstance(s, str) and skilllib.get_skill(s)]
            picked = list(dict.fromkeys(picked))[:MAX_SUB_AGENTS]
            if len(picked) < 2:
                reason = "有效的 skill_id 少于 2 个"
                skipped.append((tool, reason))
                await emit({
                    "type": "tool_call", "skill_id": "main", "tool": "dispatch",
                    "title": "未执行", "detail": reason,
                })
                continue
            titles = []
            for sid in picked:
                card = skilllib.get_skill(sid) or {}
                titles.append(card.get("title") or sid)
            body = "已派出：" + "、".join(titles)
            executed.append((tool, body))
            await emit({
                "type": "tool_call", "skill_id": "main", "tool": "dispatch",
                "title": "dispatch", "detail": "、".join(titles),
                "body": str(tool.get("goal") or state.get("question") or ""),
            })
            updates.update({
                "route": "dispatch",
                "selected_skills": picked,
                "analysis_goal": str(tool.get("goal") or state.get("question") or ""),
                "dispatched": True,
            })
            continue
        if name == "report":
            text = str(tool.get("text") or "").strip()
            executed.append((tool, "已展示给用户：\n" + text))
            await emit({"type": "tool_call", "skill_id": "main", "tool": "report", "title": "report", "detail": text[:160]})
            await emit({"type": "query_answer", "text": text, "evidences": list(state.get("evidences") or []) + evidences})
            updates["reported"] = True
            updates["known_facts"] = [text]
            continue
        body = run_readonly(tool)
        executed.append((tool, body))
        await emit({
            "type": "tool_call", "skill_id": "main", "tool": name,
            "title": name, "detail": label(tool)[:180], "body": body,
        })
        if name == "resolve_certified_metric":
            got = formula_of(body)
            if got:
                updates["metric_ready"] = True
                formula = got
                updates["metric_formula"] = got
        elif name == "read_sql_plan" and plan_hit(body):
            updates["plan_ready"] = True
        elif name == "describe_schema" and schema_hit(body):
            updates["schema_ready"] = True
        elif name == "list_skills":
            updates["skills_listed"] = True
    messages = list(state["messages"])
    messages.extend(_tool_results(decisions, executed, skipped))
    return {"messages": messages, "evidences": evidences, **updates}


def _tool_results(decisions: list[dict], executed: list[tuple[dict, str]], skipped: list[tuple[dict, str]]) -> list[dict]:
    """每个官方 tool_call 都要有一条对应结果，包括未执行的。"""
    bodies = {id(tool): body for tool, body in executed}
    reasons = {id(tool): reason for tool, reason in skipped}
    rows = []
    any_skipped = False
    for item in decisions:
        tool = item["tool"]
        call_id = str(tool.get("call_id") or "")
        if id(tool) in bodies:
            text = bodies[id(tool)]
        else:
            any_skipped = True
            text = "未执行：" + (reasons.get(id(tool)) or item.get("reason") or "未执行")
        rows.append([call_id, text])
    if rows and any_skipped:
        rows[-1][1] += "\n请在下一次回复里继续调用未执行的这些工具。只根据这些返回决定下一步。一次回复最多 4 个工具。"
    return [tool_message(call_id, text) for call_id, text in rows]


def dispatch_observation(subs: list[dict]) -> str:
    blocks = "\n\n".join(
        f"【{s.get('skill_name') or s.get('skill_id')}】({s.get('status')})\n{s.get('conclusion') or ''}"
        for s in subs
    )
    return (
        "工具 dispatch 的返回：\n"
        + blocks
        + "\n这些是专家刚刚返回的结论。下一步调用 finish，根据这些结论写 summary、consensus、divergences。\n"
        "divergences 只写专家结论互相冲突的地方。未覆盖或证据不足不要写。"
    )
