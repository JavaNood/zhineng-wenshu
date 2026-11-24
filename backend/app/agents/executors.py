"""问数点与单个 Skill 子 Agent：先查本体，再跑 ReAct 子图。"""
from __future__ import annotations
from ..ontology import loader as onto
from ..schemas import Evidence, SubResult
from ..tools import skill_tools, sql_tool
from ..skills import loader as skilllib
from .react_graph import QUERY_SYS, WORKER_SYS, run_react
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
                    "title": "查阅本体 · 对齐指标口径",
                    "detail": f"{hit['name']}（{hit['code']}）= {expr}；单位 {hit.get('unit', '')}",
                    "body": body})
    ds = onto.load_ontology().get("datasets", [])
    rels = onto.get_relationships()
    body_ont = ("数据表：\n" + "\n".join(f"- {d['source']}" for d in ds)
                + "\n允许 join 白名单：\n" + "\n".join(f"- {r['from']} → {r['to']}" for r in rels))
    await emit({"type": "tool_call", "skill_id": skill_id, "tool": "read_ontology",
                "title": "查阅本体 · 表结构与允许关联",
                "detail": f"已读取 {len(ds)} 张数据表与 join 白名单，仅允许白名单内关联",
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
    await emit_ontology(emit, skill_id, goal, question)

    skill_card_text = skill_tools.render_skill_card(card)
    known_block = (f"\n【主 Agent 已查明的总量事实，可直接引用，无需重复聚合总量】\n{known_facts}\n"
                   if known_facts else "")
    prompt_text = (
        f"【原始问题】{question}\n【分析目标】{goal}\n"
        f"【你的专家方法（Skill）】\n{skill_card_text}\n"
        + known_block +
        "【任务要求】以本技能的视角独立取数诊断，只基于真实 SQL 结果下结论；"
        "总量/口径已由主 Agent 查明，直接引用，把 SQL 用在下钻拆分（渠道/门店/类目/客群/时序等）；"
        "不与其他专家共用结论，完成后给出本视角的主要结论。")
    await emit({"type": "agent_prompt", "skill_id": skill_id, "prompt": prompt_text})

    ontology_brief = _ontology_brief()
    context_parts: list[str] = [
        f"原始问题：{question}", f"分析目标：{goal}",
        f"【本体标准指标口径，必须遵守，禁止自造】\n{ontology_brief}",
        f"{skill_tools.render_skill_card(card)}"]
    if known_facts:
        context_parts.append(
            f"【主 Agent 已查明的总量事实，可直接引用，禁止重复聚合同一总量】\n{known_facts}")
    messages = [
        {"role": "system", "content": WORKER_SYS.format(data_map=sql_tool.data_map_text())},
        {"role": "user", "content": "\n".join(context_parts)},
    ]
    final = await _invoke(
        messages, model, skill_id, skill_name, goal, "worker", emit, cursor, context)
    return _result_from_state(skill_id, skill_name, final)
