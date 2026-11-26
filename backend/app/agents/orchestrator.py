"""LangGraph 主图：主 Agent 一轮一个工具，分析专家用 Send 并行，并行度 ≤ 3。

不挂 checkpoint。单次对话的状态只活在这一次 ainvoke 里。
"""
from __future__ import annotations
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from ..config import MAX_PARALLEL
from ..schemas import IntentPoint
from ..skills import loader as skilllib
from ..tools import sql_tool
from . import executors
from .main_react import (
    dispatch_observation,
    initial_messages,
    main_turn,
    nudge_node,
    tools_node,
)
from .runtime import GraphContext


class OrchestratorState(TypedDict):
    question: str
    model: str
    messages: list
    step: int
    route: str
    pending: dict
    nudge: str
    ontology_enabled: bool
    metric_ready: bool
    plan_ready: bool
    schema_ready: bool
    metric_formula: str
    skills_listed: bool
    has_sql: bool
    reported: bool
    dispatched: bool
    why_nudged: bool
    selected_skills: list
    analysis_goal: str
    known_facts: Annotated[list[str], operator.add]
    evidences: Annotated[list[dict], operator.add]
    worker_packets: Annotated[list[dict], operator.add]
    packet_cursor: int


def build_groups(points: list[IntentPoint], question: str) -> list[dict]:
    """连续问数点合并成一次取数；chat / analysis 各自独立成组。测试与回放仍用得到。"""
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


def _initial(question: str, model: str, ontology_enabled: bool = True) -> OrchestratorState:
    return {
        "question": question,
        "model": model,
        "messages": initial_messages(question, ontology_enabled),
        "step": 0,
        "route": "",
        "pending": [],
        "nudge": "",
        "ontology_enabled": ontology_enabled,
        "metric_ready": False,
        "plan_ready": False,
        "schema_ready": False,
        "metric_formula": "",
        "skills_listed": False,
        "has_sql": False,
        "reported": False,
        "dispatched": False,
        "why_nudged": False,
        "selected_skills": [],
        "analysis_goal": "",
        "known_facts": [],
        "evidences": [],
        "worker_packets": [],
        "packet_cursor": 0,
    }


def fan_out(state: OrchestratorState):
    skills = state.get("selected_skills") or []
    if not skills:
        return "collect_workers"
    known = "\n".join(state.get("known_facts") or [])
    goal = state.get("analysis_goal") or state["question"]
    return [
        Send("worker", {
            "skill_id": sid,
            "goal": goal,
            "question": state["question"],
            "model": state["model"],
            "known_facts_text": known,
            "ontology_enabled": bool(state.get("ontology_enabled", True)),
            "ord": i,
        })
        for i, sid in enumerate(skills)
    ]


def after_main(state: OrchestratorState):
    route = state.get("route") or "finish"
    if route == "stop":
        return "finish"
    if route in ("tools", "nudge"):
        return route
    return "finish"


def after_tools(state: OrchestratorState):
    if state.get("route") == "dispatch":
        return fan_out(state)
    return "main"


async def worker_node(state: dict, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    sid = state["skill_id"]
    try:
        sname = next((s["name"] for s in skilllib.list_skills() if s["id"] == sid), sid)
        await emit({"type": "agent_start", "skill_id": sid, "skill_name": sname})
        cursor = sql_tool.new_cursor()
        sub, usage = await executors.run_skill_worker(
            sid, state["goal"], state["question"], state["model"], emit, cursor,
            known_facts=state.get("known_facts_text") or "",
            ontology_enabled=bool(state.get("ontology_enabled", True)),
            context=runtime.context,
        )
        packet = {"sub": sub.to_dict(), "usage": usage, "ord": state.get("ord", 0)}
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
        return {"worker_packets": [{"sub": sub, "usage": {}, "ord": state.get("ord", 0)}]}


async def collect_workers(state: OrchestratorState, runtime: Runtime) -> dict:
    emit = runtime.context["emit"]
    session = runtime.context["session"]
    start = state.get("packet_cursor") or 0
    batch = sorted(state["worker_packets"][start:], key=lambda p: p.get("ord", 0))
    subs = []
    for pkt in batch:
        session.add_tokens(pkt.get("usage") or {})
        session.add_sub_result(pkt["sub"])
        await emit({"type": "sub_done", "sub": pkt["sub"]})
        subs.append(pkt["sub"])
    messages = list(state["messages"]) + [{
        "role": "user",
        "content": dispatch_observation(subs),
    }]
    return {
        "packet_cursor": len(state["worker_packets"]),
        "messages": messages,
    }


async def finish_node(state: OrchestratorState, runtime: Runtime) -> dict:
    del state
    await runtime.context["emit"]({"type": "tokens", "tokens": runtime.context["session"].tokens})
    return {}


def _build():
    g = StateGraph(OrchestratorState, context_schema=GraphContext)
    g.add_node("main", main_turn)
    g.add_node("tools", tools_node)
    g.add_node("nudge", nudge_node)
    g.add_node("worker", worker_node)
    g.add_node("collect_workers", collect_workers)
    g.add_node("finish", finish_node)
    g.add_edge(START, "main")
    g.add_conditional_edges("main", after_main, ["tools", "nudge", "finish"])
    g.add_conditional_edges("tools", after_tools, ["main", "worker", "collect_workers"])
    g.add_edge("nudge", "main")
    g.add_edge("worker", "collect_workers")
    g.add_edge("collect_workers", "main")
    g.add_edge("finish", END)
    return g.compile(name="qa-orchestrator")


GRAPH = _build()

# 主循环每轮最多两个超步，再加上最多 3 个专家；真正的 60 步在专家计划执行里。
OUTER_RECURSION_LIMIT = 200


async def orchestrate(question: str, model: str, emit, session) -> None:
    enabled = bool(getattr(session, "ontology_enabled", True))
    await GRAPH.ainvoke(
        _initial(question, model, enabled),
        context={"emit": emit, "session": session},
        config={
            "max_concurrency": MAX_PARALLEL,
            "recursion_limit": OUTER_RECURSION_LIMIT,
        },
    )
