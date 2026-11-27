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

_BATCH_RULES = f"""
用接口提供的函数调用工具。不要在正文里写工具名、参数、tool_use 或 JSON。
一次回复最多调用 4 个函数。只调用一个时也用函数调用。超过 4 个时，多出来的这一轮不会执行，需要在看到结果后的下一次回复里继续调用。
互不依赖的查询可以放在同一次回复里。写 确认本体口径、确认业务指标口径、写SQL、向用户报告、查看专家、派工、收口，都要分成不同的回复：先看见上一步的结果，再做下一步。只问数值、没有要求原因或诊断时，report 成功后本轮结束，不要再调用 finish。
回复分三路，不要混用。思考只放内部推理。正文是给用户看的话，只用一两句中文说明这一步要做什么；没有要补充的话时正文留空。函数参数只放在函数调用里。不要写「分析已完成」「没有剩余步骤」这类执行状态。
专家结论返回之后，正文必须有一句衔接，告诉用户这几位专家已经分析完成，接下来由你汇总。人数按实际返回的专家数。这句写在正文，不要只写在思考里。summary、consensus、divergences 只放在 finish 的参数里。
本体开启时，开始查数的那一句写成阅读本体内容和业务语义，不要点出具体指标名或口径公式。
派专家的那一句说明为什么是选择这几位，用 list_skills 返回的 name。
查询和报告只覆盖用户点名的指标、时间和维度。用户没点名的渠道、订单状态、门店、商品、客群等，不要写进 SQL，也不要写进 report 或 finish。
用户要原因、为什么、变化、诊断时，先只报告点名的数值，若存在比较状态，可对相应的指标进行百分比计算说明。解释交给专家：report 成功后再 list_skills、再 dispatch。不要用额外 SQL 自己拆解。
向用户报告的数值指标，需要在正文说完后,说明你本次报告指标的计算口径，例如：2025年销售额xx元。口径：[你查到的计算口径]，只写指标的计算口径，不要写其他日期、过滤等附带说明。
在分析过程中，或者报告为分析的结论，或者前面已经向用户提到过该口径，则不需要重复说明口径。
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
- list_skills()：用户还要解释原因、诊断或对比时，在 report 成功之后的下一次回复单独调用。返回里如果写明某位是本会话用户融合的专家，问题同样适合多位专家时优先考虑他。不相关就不要为了优先硬选。
- dispatch(skills, goal)：从刚看过的技能里选。一轮最多 3 位。没有本会话融合专家时至少 2 位；选了融合专家时可以只派他 1 位，也可以再加其他专家，总数不超过 3。不要把融合专家和他的父专家或更上层专家派在同一轮。没调用过 list_skills 不要派。有 SQL 结果但还没 report 时不要派。
  goal 只写全体专家共用的问题：用户点名的指标、时间范围、要解释的变化。不要把某一位专家的拆法写进 goal，例如渠道、大区、门店、季节、促销、类目、品牌。这些由各位专家按自己的技能决定。
  给用户看的那一两句必须使用 list_skills 返回的 name，并说明每位专家各自负责什么。例如：我派三位专家并行分析：量价结构拆解专家-小王拆解订单量和客单价的贡献，时序季节诊断专家-小黄判断是否为季节或促销节奏。
- finish(summary, consensus, divergences)：结束这轮对话。寒暄、与数据无关的问题直接 finish，不要查数。
  寒暄时，正文写给用户听的那句话。是不是打招呼、为什么不查数，只写在思考里。summary 与正文相同，不要写「用户仅打招呼，已回应并说明」这类内部说明。
  只问了数值时，report 成功后就结束，不要再调用 finish，也不要在报告后再写一段收口说明。
  派过专家之后，必须等专家结论返回再 finish。调用 finish 时，正文写一句衔接，例如「3位专家已经分析完成，我来汇总结论」。这句必须出现在正文，不要只放在思考里，也不要把 summary 写进正文。
  summary 先写各方一致的结论，几句即可，不要把专家原文再贴一遍。至少两个专家共同指向的写进 consensus，每条一句。不一致的写进 divergences，每条一句，point 里点名专家的 name（title），不要写 skill_id。没有分歧就给空数组。不要编造专家没说过的判断。
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


def initial_messages(question: str, ontology_enabled: bool = True, session_id: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": main_sys(ontology_enabled)},
        {"role": "user", "content": question},
    ]
    catalog = _fused_catalog(session_id)
    if catalog:
        messages.append({"role": "user", "content": catalog})
    return messages


def _fused_catalog(session_id: str | None) -> str:
    if not session_id:
        return ""
    rows = [item for item in skilllib.list_skills(session_id) if item.get("scope") == "session"]
    if not rows:
        return ""
    lines = [f"- {item['id']}｜{item['name']}：{item['description']}" for item in rows]
    return "本会话已有的融合专家。分析问题时与基础专家一起考虑：\n" + "\n".join(lines)


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
                "nudge": "工具没有执行：这个问题在问原因或诊断，还没有派出专家。请先 list_skills，再 dispatch。没有本会话融合专家时至少 2 位；有融合专家时可以只派他 1 位。不要直接 finish。",
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
        # 正文已经展示过。寒暄的 summary 只在没有正文时才当作回复，避免把内部判断再印一遍。
        if not (content or "").strip() and summary:
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


def reject_dispatch(picked: list[str], session_id: str | None) -> str | None:
    """人数和祖先互斥。通过时返回 None。融合专家的 ancestors 含父专家和更上层。"""
    cards = []
    for sid in picked[:MAX_SUB_AGENTS]:
        card = skilllib.get_skill(sid, session_id)
        if card:
            cards.append(card)
    if not cards:
        return "有效的 skill_id 少于 2 个"
    fused = [card for card in cards if card.get("scope") == "session"]
    if not fused and len(cards) < 2:
        return "有效的 skill_id 少于 2 个"
    picked_ids = {card["id"] for card in cards}
    parts = []
    for card in cards:
        if card.get("scope") != "session":
            continue
        hit = [aid for aid in (card.get("ancestors") or []) if aid in picked_ids]
        if not hit:
            continue
        names = []
        for aid in hit:
            other = skilllib.get_skill(aid, session_id) or {}
            names.append(other.get("title") or aid)
        who = card.get("title") or card["id"]
        parts.append(f"{who} 与 {'、'.join(names)} 不能同时出场")
    if not parts:
        return None
    return "；".join(parts) + "。请去掉冲突的专家后重新 dispatch。"


async def tools_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    session_id = getattr(session, "session_id", None)
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
            raw = [s for s in (tool.get("skills") or []) if isinstance(s, str)]
            picked = []
            for sid in raw:
                if sid in picked:
                    continue
                if skilllib.get_skill(sid, session_id):
                    picked.append(sid)
                if len(picked) >= MAX_SUB_AGENTS:
                    break
            reason = reject_dispatch(picked, session_id)
            if reason:
                skipped.append((tool, reason))
                await emit({
                    "type": "tool_call", "skill_id": "main", "tool": "dispatch",
                    "title": "未执行", "detail": reason,
                })
                continue
            titles = []
            for sid in picked:
                card = skilllib.get_skill(sid, session_id) or {}
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
            if updates.get("route") != "dispatch" and not _WHY.search(str(state.get("question") or "")):
                updates["route"] = "stop"
            continue
        body = run_readonly(tool, session_id)
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
        + "\n这些是专家刚刚返回的结论。\n"
        "正文写一句衔接，告诉用户这几位专家已经分析完成，接下来由你汇总。人数用上面实际返回的专家人数。\n"
        "这句必须写在正文，不要只写在思考里。summary、consensus、divergences 只放在 finish 的参数里。\n"
        "然后调用 finish。summary 先写各方一致的结论，几句即可，不要把专家原文再贴一遍。\n"
        "consensus 只收至少两位共同指向的判断，每条一句。\n"
        "divergences 只写互相冲突的判断，每条一句，point 里点名专家的 name。没有分歧就给空数组。未覆盖或证据不足不要写。\n"
    )
