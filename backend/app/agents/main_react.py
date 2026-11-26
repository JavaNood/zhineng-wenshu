"""主 Agent 的 ReAct：每一轮只观察已有结果，再决定一个工具。

思考文案和工具参数都来自当次模型输出。这里只规定工具协议和执行顺序。
"""
from __future__ import annotations
import re

from langgraph.runtime import Runtime

from .. import llm
from ..config import MAX_SUB_AGENTS, RESULT_ROWS
from ..schemas import Evidence, FinalResult
from ..skills import loader as skilllib
from ..tools import ontology_tools, skill_tools, sql_tool
from .textutil import _extract_json, _rows_preview

# 主循环步数。需求里单个 Agent 不超过 60，这里留出 SQL 重试，避免一轮问题打满。
MAIN_MAX_TURNS = 16

_WHY = re.compile(r"原因|为什么|为何|下降|诊断|异常")

MAIN_SYS = """你是智能问数的主 Agent，用 ReAct 推进：观察已经拿到的结果，决定下一步，只调用一个工具。
不要在同一轮里把后面的取数、派专家、汇总提前说完。

每一轮的 content 只有两部分：
1. 工具块之前用一两句中文，只说明这一步要做什么。这两句会直接展示给用户。
2. 紧接着只输出一个 ```tool 代码块，里面是 JSON。

工具：
- {"tool":"read_ontology","metric":"指标名"}
  阅读本体里的指标口径、表字段和允许关联。任何 SQL 之前必须先调用。
- {"tool":"run_sql","sql":"一条 DuckDB 只读 SELECT"}
  FROM/JOIN 用本体里的 sql 字段，不要用 source。列用中文原名。多个时间、多个指标用一条 SQL 取回。没读过本体不要调用。
- {"tool":"report","text":"给用户看的数值答案"}
  已经有 SQL 结果、需要先把数值告诉用户时调用。text 里的数字只能来自工具返回。
- {"tool":"list_skills"}
  查看可派的分析专家。用户要解释原因、诊断或对比时，在总量取到之后调用。
- {"tool":"dispatch","skills":["skill_id","skill_id"],"goal":"这一次要专家回答的问题"}
  从刚看过的技能里选 2 到 3 个，派它们并行分析。没调用过 list_skills 不要派。
- {"tool":"finish","summary":"整体结论","consensus":[{"point":"判断","support":["skill_id"]}],"divergences":[{"point":"分歧","why":"原因","pending":"待复核"}]}
  结束这轮对话。寒暄、与数据无关的问题直接 finish，不要查数。
  只问了数值并且已经 report 过，可以 finish，summary 用已经报告的数值。
  派过专家之后，必须等专家结论返回再 finish：至少两个专家共同指向的写进 consensus，互相冲突的写进 divergences，没有冲突就给空数组。不要编造专家没说过的判断。
  未覆盖、证据不足、某步没取成，不要写进 summary、consensus、divergences。

禁止编造数字和指标口径。一轮一个工具。"""


def _preface(content: str) -> str:
    """工具块前面的一两句，是模型写给用户的这一步，不截成固定话术。"""
    text = content or ""
    cut = text.find("```")
    if cut >= 0:
        text = text[:cut]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:2])[:300]


def parse_tool(content: str) -> dict | None:
    """从模型这一轮的 content 里取唯一的工具。兼容它直接写 ```sql 的情况。"""
    block = re.search(r"```tool\s*([\s\S]*?)```", content or "", re.I)
    if block:
        data = _extract_json(block.group(1))
        if isinstance(data, dict) and data.get("tool"):
            return data
    sql = re.search(r"```sql\s*([\s\S]*?)```", content or "", re.I)
    if sql and sql.group(1).strip():
        return {"tool": "run_sql", "sql": sql.group(1).strip()}
    data = _extract_json(content or "")
    if isinstance(data, dict) and data.get("tool"):
        return data
    if isinstance(data, dict) and data.get("summary") and "```sql" not in (content or ""):
        return {"tool": "finish", **data}
    return None


def ontology_payload(metric: str) -> str:
    metric = (metric or "").strip()
    parts = [ontology_tools.read_ontology(metric)]
    if metric:
        parts.append(ontology_tools.resolve_metric(metric))
    return "\n\n".join(parts)


def initial_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": MAIN_SYS},
        {"role": "user", "content": question},
    ]


def _add_usage(state_usage_ignored, resp_usage: dict, session) -> None:
    del state_usage_ignored
    session.add_tokens(resp_usage or {})


async def main_turn(state: dict, runtime: Runtime) -> dict:
    """一轮模型调用：思维链和工具前的句子随 token 推送，结束后再解析一个工具。"""
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    step = int(state.get("step") or 0) + 1
    if step > MAIN_MAX_TURNS:
        await emit({"type": "error", "message": "主 Agent 达到步数上限，已停止。"})
        return {"step": step, "route": "stop"}

    held = ""
    closed = False
    streamed_act = False

    async def _on_reasoning(text: str) -> None:
        if not text:
            return
        await emit({"type": "reasoning", "skill_id": "main", "text": text})

    async def _on_content(text: str) -> None:
        """工具块之前的句子边生成边推。``` 之后是工具 JSON，不再展示。"""
        nonlocal held, closed, streamed_act
        if closed or not text:
            return
        chunk = held + text
        cut = chunk.find("```")
        if cut >= 0:
            show = chunk[:cut]
            held = ""
            closed = True
            if show:
                streamed_act = True
                await emit({"type": "act", "skill_id": "main", "text": show})
            return
        if len(chunk) <= 2:
            held = chunk
            return
        show, held = chunk[:-2], chunk[-2:]
        streamed_act = True
        await emit({"type": "act", "skill_id": "main", "text": show})

    resp = await llm.chat_stream(
        state["messages"], model=state["model"],
        on_reasoning=_on_reasoning, on_content=_on_content,
    )
    _add_usage(None, resp.get("usage") or {}, session)
    content = resp.get("content") or ""
    messages = list(state["messages"]) + [{"role": "assistant", "content": content}]
    if not closed and held.strip():
        streamed_act = True
        await emit({"type": "act", "skill_id": "main", "text": held})
    preface = _preface(content)
    # 测试桩不走 on_content。真实流式已经推过的，不再整段补发。
    if preface and not streamed_act:
        await emit({"type": "act", "skill_id": "main", "text": preface})

    tool = parse_tool(content) or {}
    name = str(tool.get("tool") or "").strip()
    question = state["question"]
    base = {
        "messages": messages,
        "step": step,
        "pending": tool,
        "nudge": "",
    }

    if name == "read_ontology":
        return {**base, "route": "ontology"}
    if name == "list_skills":
        return {**base, "route": "skills"}
    if name == "run_sql":
        if not state.get("ontology_read"):
            return {**base, "route": "nudge", "nudge": "工具没有执行：还没有 read_ontology，不能 run_sql。请先调用 read_ontology。"}
        if not str(tool.get("sql") or "").strip():
            return {**base, "route": "nudge", "nudge": "工具没有执行：run_sql 缺少 sql。请重新输出带 SQL 的 run_sql。"}
        return {**base, "route": "sql"}
    if name == "report":
        if not state.get("has_sql"):
            return {**base, "route": "nudge", "nudge": "工具没有执行：还没有成功的 SQL 结果，不能 report。请先 run_sql。"}
        if not str(tool.get("text") or "").strip():
            return {**base, "route": "nudge", "nudge": "工具没有执行：report 缺少 text。"}
        return {**base, "route": "report"}
    if name == "dispatch":
        if not state.get("skills_listed"):
            return {**base, "route": "nudge", "nudge": "工具没有执行：还没有 list_skills，不能 dispatch。请先调用 list_skills。"}
        if state.get("dispatched"):
            return {**base, "route": "nudge", "nudge": "工具没有执行：这轮已经派过专家。请根据专家结论调用 finish。"}
        picked = [s for s in (tool.get("skills") or []) if isinstance(s, str) and skilllib.get_skill(s)]
        picked = list(dict.fromkeys(picked))[:MAX_SUB_AGENTS]
        if len(picked) < 2:
            return {**base, "route": "nudge", "nudge": "工具没有执行：dispatch 里有效的 skill_id 少于 2 个。请按 list_skills 的返回重新选择至少 2 个。"}
        chosen = []
        for sid in picked:
            card = skilllib.get_skill(sid) or {}
            chosen.append(card.get("name") or sid)
        await emit({
            "type": "tool_call",
            "skill_id": "main",
            "tool": "dispatch",
            "title": "dispatch",
            "detail": "、".join(chosen),
            "body": str(tool.get("goal") or question),
        })
        return {
            **base,
            "route": "dispatch",
            "selected_skills": picked,
            "analysis_goal": str(tool.get("goal") or question),
            "dispatched": True,
        }
    if name == "finish":
        if _WHY.search(question) and not state.get("dispatched") and not state.get("why_nudged"):
            return {
                **base,
                "route": "nudge",
                "why_nudged": True,
                "nudge": "工具没有执行：这个问题在问原因或诊断，还没有派出专家。请先 list_skills，再 dispatch 至少 2 个技能，不要直接 finish。",
            }
        await _emit_finish(state, tool, preface, content, emit, session)
        return {**base, "route": "stop"}
    return {**base, "route": "nudge", "nudge": "没有解析到一个 ```tool JSON。请只输出一个工具块。"}


async def _emit_finish(state, tool, preface, content, emit, session) -> None:
    summary = str(tool.get("summary") or preface or "").strip()
    if not summary:
        summary = (content or "").strip()
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
        # evidences 在事件里用 dict；FinalResult 只收 Evidence
        final.evidences = []
        payload = final.to_dict()
        payload["evidences"] = evidences
        session.set_final(payload)
        await emit({"type": "consensus", "consensus": consensus, "divergences": divergences})
        await emit({"type": "final", "summary": summary, "evidences": evidences})
        return
    if not state.get("reported") and summary:
        await emit({"type": "query_answer", "text": summary, "evidences": evidences})


_REAL_TOOLS = {"read_ontology", "run_sql", "report", "list_skills", "dispatch", "finish"}


async def nudge_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    note = state.get("nudge") or "请换一个工具。"
    pending = state.get("pending") or {}
    name = str(pending.get("tool") or "").strip()
    # 没解析出工具时，这句话只回给模型重试，界面上不先画一个工具。
    if name in _REAL_TOOLS:
        await emit({
            "type": "tool_call",
            "skill_id": "main",
            "tool": name,
            "title": name,
            "detail": note,
        })
    messages = list(state["messages"]) + [{"role": "user", "content": note}]
    return {"messages": messages}


async def ontology_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    metric = str((state.get("pending") or {}).get("metric") or "")
    body = ontology_payload(metric)
    detail = metric or "全部标准指标"
    await emit({
        "type": "tool_call",
        "skill_id": "main",
        "tool": "read_ontology",
        "title": "read_ontology",
        "detail": detail,
        "body": body,
    })
    messages = list(state["messages"]) + [{
        "role": "user",
        "content": "工具 read_ontology 的返回：\n" + body + "\n只根据这个返回决定下一步，调用一个工具。",
    }]
    return {"messages": messages, "ontology_read": True}


async def skills_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    body = skill_tools.list_skills()
    catalog = skilllib.list_skills()
    await emit({
        "type": "tool_call",
        "skill_id": "main",
        "tool": "list_skills",
        "title": "list_skills",
        "detail": f"{len(catalog)} 个技能",
        "body": body,
    })
    messages = list(state["messages"]) + [{
        "role": "user",
        "content": "工具 list_skills 的返回：\n" + body + "\n只根据这个返回决定下一步，调用一个工具。",
    }]
    return {"messages": messages, "skills_listed": True}


async def sql_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    sql = str((state.get("pending") or {}).get("sql") or "")
    res = sql_tool.run_sql(sql, cursor=sql_tool.new_cursor())
    if res.get("error"):
        detail = res["error"]
        preview = detail
        flag = {}
    else:
        detail = f"返回 {res['row_count']} 行"
        shown = res["rows"][:RESULT_ROWS]
        preview = f"结果（{res['row_count']} 行）：\n{_rows_preview(shown, res['row_count'])}"
        ev = Evidence(metric=state.get("question") or "", sql=res["sql"],
                      row_count=res["row_count"], rows=shown)
        flag = {"has_sql": True, "evidences": [ev.to_dict()]}
    await emit({
        "type": "tool_call",
        "skill_id": "main",
        "tool": "run_sql",
        "title": "run_sql",
        "detail": detail,
        "sql": res.get("sql") or sql,
    })
    messages = list(state["messages"]) + [{
        "role": "user",
        "content": "工具 run_sql 的返回：\n" + preview + "\n只根据这个返回决定下一步，调用一个工具。",
    }]
    return {"messages": messages, **flag}


async def report_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    text = str((state.get("pending") or {}).get("text") or "").strip()
    await emit({
        "type": "tool_call",
        "skill_id": "main",
        "tool": "report",
        "title": "report",
        "detail": text[:160],
    })
    await emit({
        "type": "query_answer",
        "text": text,
        "evidences": list(state.get("evidences") or []),
    })
    messages = list(state["messages"]) + [{
        "role": "user",
        "content": "工具 report 的返回：已把下面这段展示给用户。\n" + text + "\n只根据目前的结果决定下一步，调用一个工具。",
    }]
    return {"messages": messages, "reported": True, "known_facts": [text]}


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
