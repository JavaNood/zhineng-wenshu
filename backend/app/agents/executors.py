"""问数点仍可走 ReAct 子图。分析专家先计划，再在执行里调用共享查数工具。"""
from __future__ import annotations
from ..schemas import Evidence, SubResult
from ..skills import loader as skilllib
from .plan_execute import run_plan_execute
from .react_graph import QUERY_SYS, run_react
from .runtime import GraphContext, bind_cursor, reset_cursor


def _result_from_state(skill_id: str, skill_name: str, final: dict) -> tuple[SubResult, dict]:
    evidences = [
        Evidence(
            metric=e.get("metric", ""),
            formula=e.get("formula", ""),
            sql=e.get("sql", ""),
            window=e.get("window", ""),
            rows=list(e.get("rows") or []),
            row_count=int(e.get("row_count") or 0),
        )
        for e in (final.get("evidences") or [])
    ]
    sub = SubResult(
        skill_id=skill_id,
        skill_name=skill_name,
        conclusion=final.get("conclusion") or "",
        evidences=evidences,
        status=final.get("status") or "done",
    )
    usage = final.get("usage") or {"prompt": 0, "completion": 0, "total": 0}
    return sub, usage


async def _invoke(messages, model, skill_id, skill_name, goal, profile, emit, cursor, context):
    ctx: GraphContext = context or {"emit": emit, "session": None}
    token = bind_cursor(cursor)
    try:
        return await run_react(
            messages=messages, model=model, skill_id=skill_id, skill_name=skill_name,
            goal=goal, profile=profile, context=ctx,
        )
    finally:
        reset_cursor(token)


async def run_query_point(goal: str, question: str, model: str, emit, cursor, context=None):
    local_fallback = {"prompt": 0, "completion": 0, "total": 0}
    messages = [
        {"role": "system", "content": QUERY_SYS},
        {"role": "user", "content": f"原始问题：{question}\n查询目标：{goal}"},
    ]
    final = await _invoke(
        messages, model, "query", "问数执行", goal, "query", emit, cursor, context)
    sub, usage = _result_from_state("query", "问数执行", final)
    return sub, usage or local_fallback


async def run_skill_worker(
    skill_id: str,
    goal: str,
    question: str,
    model: str,
    emit,
    cursor,
    known_facts: str = "",
    ontology_enabled: bool = True,
    context=None,
    session_id: str | None = None,
):
    card = skilllib.get_skill(skill_id, session_id)
    local_usage = {"prompt": 0, "completion": 0, "total": 0}
    if not card:
        return SubResult(skill_id=skill_id, skill_name=skill_id,
                         conclusion=f"未找到 Skill {skill_id}", status="error"), local_usage

    skill_name = card.get("title", card["name"])
    return await run_plan_execute(
        skill_id=skill_id,
        skill_name=skill_name,
        goal=goal,
        question=question,
        model=model,
        emit=emit,
        cursor=cursor,
        known_facts=known_facts,
        card=card,
        ontology_enabled=ontology_enabled,
    )
