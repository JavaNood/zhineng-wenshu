"""问数点仍可走 ReAct 子图。分析专家改为先计划、再按计划执行。"""
from __future__ import annotations
from ..ontology import loader as onto
from ..schemas import Evidence, SubResult
from ..tools import sql_tool
from ..skills import loader as skilllib
from .plan_execute import run_plan_execute
from .react_graph import QUERY_SYS, run_react
from .runtime import GraphContext, bind_cursor, reset_cursor


def _ontology_brief() -> str:
    return "\n".join(
        f"- {m['name']}({m['code']} = {m['expression'] or m['formula']})"
        for m in onto.list_metrics())


async def emit_ontology(emit, skill_id: str, goal: str, question: str) -> None:
    """显式查阅本体：先对齐指标口径，再确认表结构与允许关联。"""
    hit = onto.resolve_metric(goal) or onto.resolve_metric(question)
    if hit:
        expr = hit.get("expression") or hit.get("formula")
        dims = hit.get("dims") or []
        dims_txt = "、".join(dims) if dims else "未限定"
        body = (f"指标：{hit['name']}（{hit['code']}）\n口径：{expr}\n"
                f"单位：{hit.get('unit', '')}\n可下钻维度：{dims_txt}")
        await emit({"type": "tool_call", "skill_id": skill_id, "tool": "resolve_metric",
                    "title": f"{hit['name']}（{hit['code']}）",
                    "detail": f"{expr}；单位 {hit.get('unit', '')}",
                    "body": body})
    ds = onto.load_ontology().get("datasets", [])
    rels = onto.get_relationships()
    body_ont = ("数据表：\n" + "\n".join(f"- {d['source']}" for d in ds)
                + "\n允许 join 白名单：\n" + "\n".join(f"- {r['from']} → {r['to']}" for r in rels))
    await emit({"type": "tool_call", "skill_id": skill_id, "tool": "read_ontology",
                "title": "read_ontology",
                "detail": f"{len(ds)} 张表，{len(rels)} 条允许关联",
                "body": body_ont})


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
    ontology_brief = _ontology_brief()
    await emit_ontology(emit, "query", goal, question)
    messages = [
        {"role": "system", "content": QUERY_SYS.format(data_map=sql_tool.data_map_text())},
        {"role": "user", "content": (
            f"【本体标准指标口径，必须遵守，禁止自造】\n{ontology_brief}\n\n"
            f"原始问题：{question}\n查询目标：{goal}")},
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
    context=None,
):
    card = skilllib.get_skill(skill_id)
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
    )
