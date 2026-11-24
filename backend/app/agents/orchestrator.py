"""LangGraph 主图：意图规划 → 按问题点路由 → 问数 / 寒暄 / 分析扇出 → 汇聚。

不挂 checkpoint。单次对话的状态只活在这一次 ainvoke 里，会话快照仍由 session JSON 负责。
分析子 Agent 用 Send 扇出，并行度由运行配置 max_concurrency 限制为 3。
"""
from __future__ import annotations
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from .. import llm
from ..config import MAX_PARALLEL, MAX_SUB_AGENTS
from ..schemas import IntentPoint
from ..skills import loader as skilllib
from ..tools import sql_tool
from . import executors, planner
from .merge import merge_results, sub_from_dict
from .runtime import GraphContext


class OrchestratorState(TypedDict):
    question: str
    model: str
    groups: list[dict]
    group_index: int
    known_facts: Annotated[list[str], operator.add]
    worker_packets: Annotated[list[dict], operator.add]
    packet_cursor: int
    selected_skills: list[str]


def build_groups(points: list[IntentPoint], question: str) -> list[dict]:
    """连续问数点合并成一次取数；chat / analysis 各自独立成组。"""
    groups: list[dict] = []
    qbuf: list[IntentPoint] = []

    def flush_query() -> None:
        if not qbuf:
            return
        merged_goal = "；".join(dict.fromkeys(
            m.plan.goal for m in qbuf if m.plan and m.plan.goal)) or question
        groups.append({"kind": "query", "goal": merged_goal, "skill_ids": []})
        qbuf.clear()

    for p in points:
        if p.intent == "query":
            qbuf.append(p)
        else:
            flush_query()
            groups.append({
                "kind": p.intent,
                "goal": p.plan.goal if p.plan else "",
                "skill_ids": list(p.plan.skill_ids) if p.plan else [],
            })
    flush_query()
    return groups


def _initial(question: str, model: str) -> OrchestratorState:
    return {
        "question": question,
        "model": model,
        "groups": [],
        "group_index": 0,
        "known_facts": [],
        "worker_packets": [],
        "packet_cursor": 0,
        "selected_skills": [],
    }


async def plan_node(state: OrchestratorState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    question = state["question"]
    points, usage, plan_text = await planner.classify_and_plan(question, state["model"])
    session.add_tokens(usage)
    if plan_text:
        await emit({"type": "reasoning", "skill_id": "planner", "text": plan_text})
    await emit({"type": "intent", "points": [
        {"index": p.index, "text": p.raw_text, "intent": p.intent,
         "goal": p.plan.goal if p.plan else "", "skills": p.plan.skill_ids if p.plan else []}
        for p in points]})
    session.set_plan({"points": [
        {"index": p.index, "text": p.raw_text, "intent": p.intent, "goal": p.plan.goal,
         "skills": p.plan.skill_ids,
         "steps": [vars(s) for s in p.plan.steps]} for p in points]})
    await emit({"type": "plan", "plan": session.plan})
    return {"groups": build_groups(points, question), "group_index": 0}


def route_group(state: OrchestratorState) -> str:
    groups = state["groups"]
    if state["group_index"] >= len(groups):
        return "merge" if state["worker_packets"] else "finish"
    kind = groups[state["group_index"]]["kind"]
    if kind == "chat":
        return "chat"
    if kind == "query":
        return "query"
    if kind == "analysis":
        return "prepare_analysis"
    return "finish"


async def chat_node(state: OrchestratorState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    resp = await llm.chat(
        [{"role": "system", "content": "你是友好的助手，直接简短回答寒暄与一般问题，不查数。"},
         {"role": "user", "content": state["question"]}], model=state["model"])
    session.add_tokens(resp.get("usage", {}))
    await emit({"type": "chat_answer", "text": resp.get("content", "")})
    return {"group_index": state["group_index"] + 1}


async def query_node(state: OrchestratorState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    group = state["groups"][state["group_index"]]
    cursor = sql_tool.new_cursor()
    sub, usage = await executors.run_query_point(
        group["goal"], state["question"], state["model"], emit, cursor,
        context=runtime.context)
    session.add_tokens(usage)
    await emit({"type": "query_answer", "text": sub.conclusion,
                "evidences": [e.to_dict() for e in sub.evidences]})
    return {
        "group_index": state["group_index"] + 1,
        "known_facts": [sub.conclusion],
    }


async def prepare_analysis(state: OrchestratorState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    group = state["groups"][state["group_index"]]
    selected = list(group.get("skill_ids") or [])[:MAX_SUB_AGENTS]
    await emit({"type": "reasoning", "skill_id": "planner",
                "text": "数值部分我已经直接查到。要回答\"变化的原因\"，只看年度总量不足以定位，"
                        "我将进一步匹配合适的分析本领做下钻诊断；先通读各项本领，再结合问题按需挑选，不固定派齐。"})
    all_skills = skilllib.list_skills()
    skill_body = "\n".join(
        f"● {s['name']}（{s['id']}）\n"
        f"  适用场景：{s.get('trigger', '')}\n"
        f"  擅长角度：{s.get('expert_role') or s.get('description', '')}"
        for s in all_skills)
    await emit({"type": "tool_call", "skill_id": "planner", "tool": "list_skills",
                "title": "通读可用分析本领（技能库）",
                "detail": f"共 {len(all_skills)} 项分析本领可选，按问题挑选、不固定派齐",
                "body": skill_body})
    chosen = [s for s in all_skills if s["id"] in selected]
    dropped = [s for s in all_skills if s["id"] not in selected]
    pick_txt = "、".join(f"{s['name']}（{s.get('expert_role', '')}）" for s in chosen)
    drop_txt = "、".join(s["name"] for s in dropped)
    await emit({"type": "reasoning", "skill_id": "planner",
                "text": ("阅读各项本领后，结合本问题选择派出："
                         f"{pick_txt}，各自独立取数、从不同视角交叉诊断；"
                         + (f"其余本领（{drop_txt}）与本问题关联较弱，本次暂不派出；" if drop_txt else "")
                         + "完成后我再汇总各家结论、比对共识与差异。")})
    return {"selected_skills": selected}


def fan_out(state: OrchestratorState):
    skills = state["selected_skills"]
    if not skills:
        return "collect_workers"
    group = state["groups"][state["group_index"]]
    known = "\n".join(state["known_facts"])
    return [
        Send("worker", {
            "skill_id": sid,
            "goal": group["goal"],
            "question": state["question"],
            "model": state["model"],
            "known_facts_text": known,
            "group_index": state["group_index"],
            "ord": i,
        })
        for i, sid in enumerate(skills)
    ]


async def worker_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    sid = state["skill_id"]
    packet_meta = {
        "group_index": state.get("group_index", 0),
        "ord": state.get("ord", 0),
    }
    try:
        sname = next((s["name"] for s in skilllib.list_skills() if s["id"] == sid), sid)
        await emit({"type": "agent_start", "skill_id": sid, "skill_name": sname})
        cursor = sql_tool.new_cursor()
        sub, usage = await executors.run_skill_worker(
            sid, state["goal"], state["question"], state["model"], emit, cursor,
            known_facts=state.get("known_facts_text") or "",
            context=runtime.context,
        )
        packet = {"sub": sub.to_dict(), "usage": usage, **packet_meta}
        return {"worker_packets": [packet]}
    except Exception as res:
        sub = {
            "skill_id": sid,
            "skill_name": sid,
            "conclusion": f"专家 {sid} 执行异常：{type(res).__name__}: {res}",
            "status": "error",
            "note": "已隔离，不影响其他专家",
            "evidences": [],
        }
        return {"worker_packets": [{"sub": sub, "usage": {}, **packet_meta}]}


async def collect_workers(state: OrchestratorState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    start = state["packet_cursor"]
    batch = sorted(state["worker_packets"][start:], key=lambda p: p.get("ord", 0))
    for pkt in batch:
        session.add_tokens(pkt.get("usage") or {})
        session.add_sub_result(pkt["sub"])
        await emit({"type": "sub_done", "sub": pkt["sub"]})
    return {
        "packet_cursor": len(state["worker_packets"]),
        "group_index": state["group_index"] + 1,
    }


async def merge_node(state: OrchestratorState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    packets = sorted(
        state["worker_packets"],
        key=lambda p: (p.get("group_index", 0), p.get("ord", 0)),
    )
    subs = [sub_from_dict(p["sub"]) for p in packets]
    expert_subs = [s for s in subs if s.skill_id != "query"]
    if expert_subs:
        done_n = sum(1 for s in expert_subs if s.status == "done")
        await emit({"type": "reasoning", "skill_id": "planner",
                    "text": f"{done_n} 位专家已完成独立取数与诊断。现在由我来汇总各家结论、"
                            "比对共识与差异，再给出整体判断。"})
    final, usage = await merge_results(subs, state["model"])
    session.add_tokens(usage)
    session.set_final(final.to_dict())
    await emit({"type": "consensus", "consensus": final.consensus, "divergences": final.divergences})
    await emit({"type": "final", "summary": final.summary,
                "evidences": [e.to_dict() for e in final.evidences]})
    return {}


async def finish_node(state: OrchestratorState, runtime: Runtime) -> dict:
    del state
    await runtime.context["emit"]({"type": "tokens", "tokens": runtime.context["session"].tokens})
    return {}


def _build():
    g = StateGraph(OrchestratorState, context_schema=GraphContext)
    g.add_node("plan", plan_node)
    g.add_node("chat", chat_node)
    g.add_node("query", query_node)
    g.add_node("prepare_analysis", prepare_analysis)
    g.add_node("worker", worker_node)
    g.add_node("collect_workers", collect_workers)
    g.add_node("merge", merge_node)
    g.add_node("finish", finish_node)
    g.add_edge(START, "plan")
    routes = {
        "chat": "chat",
        "query": "query",
        "prepare_analysis": "prepare_analysis",
        "merge": "merge",
        "finish": "finish",
    }
    for src in ("plan", "chat", "query", "collect_workers"):
        g.add_conditional_edges(src, route_group, routes)
    g.add_conditional_edges("prepare_analysis", fan_out, ["worker", "collect_workers"])
    g.add_edge("worker", "collect_workers")
    g.add_edge("merge", "finish")
    g.add_edge("finish", END)
    return g.compile(name="qa-orchestrator")


GRAPH = _build()

# 问题点不多，主图超步远小于这个上限；真正的 60 步循环在 ReAct 子图里。
OUTER_RECURSION_LIMIT = 200


async def orchestrate(question: str, model: str, emit, session) -> None:
    await GRAPH.ainvoke(
        _initial(question, model),
        context={"emit": emit, "session": session},
        config={
            "max_concurrency": MAX_PARALLEL,
            "recursion_limit": OUTER_RECURSION_LIMIT,
        },
    )
