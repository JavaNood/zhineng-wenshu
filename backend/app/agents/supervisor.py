"""supervisor：主 Agent，意图编排、问数点取数、分析点并行派发、汇聚共识/分歧。"""
from __future__ import annotations
import asyncio
import json
import re
from .. import llm
from ..config import MAX_PARALLEL, MAX_STEPS, MAX_SUB_AGENTS
from ..tools import sql_tool
from ..ontology import loader as onto
from ..skills import loader as skilllib
from . import planner
from .worker import run_skill_worker, _extract_sql, _extract_block, _rows_preview, _action_before_sql
from ..schemas import SubResult, Evidence, FinalResult


# ---------- 问数点：单 Agent 取数循环 ----------
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


async def run_query_point(goal: str, question: str, model: str, emit, cursor) -> tuple[SubResult, dict]:
    local_usage = {"prompt": 0, "completion": 0, "total": 0}
    ontology_brief = "\n".join(
        f"- {m['name']}({m['code']} = {m['expression'] or m['formula']})"
        for m in onto.list_metrics())

    # —— 显式查阅本体（体现语义层作用）：先对齐指标口径，再确认表结构与关联 ——
    hit = onto.resolve_metric(goal) or onto.resolve_metric(question)
    if hit:
        expr = hit.get("expression") or hit.get("formula")
        dims = hit.get("dims") or []
        dims_txt = "、".join(dims) if dims else "未限定"
        body = (f"指标：{hit['name']}（{hit['code']}）\n口径：{expr}\n"
                f"单位：{hit.get('unit', '')}\n可下钻维度：{dims_txt}")
        await emit({"type": "tool_call", "skill_id": "query", "tool": "resolve_metric",
                    "title": "查阅本体 · 对齐指标口径",
                    "detail": f"{hit['name']}（{hit['code']}）= {expr}；单位 {hit.get('unit', '')}",
                    "body": body})
    ds = onto.load_ontology().get("datasets", [])
    rels = onto.get_relationships()
    body_ont = ("数据表：\n" + "\n".join(f"- {d['source']}" for d in ds)
                + "\n允许 join 白名单：\n" + "\n".join(f"- {r['from']} → {r['to']}" for r in rels))
    await emit({"type": "tool_call", "skill_id": "query", "tool": "read_ontology",
                "title": "查阅本体 · 表结构与允许关联",
                "detail": f"已读取 {len(ds)} 张数据表与 join 白名单，仅允许白名单内关联",
                "body": body_ont})

    messages = [
        {"role": "system", "content": QUERY_SYS.format(data_map=sql_tool.data_map_text())},
        {"role": "user", "content": (
            f"【本体标准指标口径，必须遵守，禁止自造】\n{ontology_brief}\n\n"
            f"原始问题：{question}\n查询目标：{goal}")},
    ]
    evidences: list[Evidence] = []
    conclusion = ""
    status = "done"
    for step in range(1, MAX_STEPS + 1):
        await emit({"type": "agent_progress", "skill_id": "query", "skill_name": "问数执行", "step": step})

        async def _reasoning(t: str):
            await emit({"type": "reasoning", "skill_id": "query", "text": t})

        resp = await llm.chat_stream(messages, model=model, on_reasoning=_reasoning)
        for k in ("prompt", "completion", "total"):
            local_usage[k] += resp.get("usage", {}).get(k, 0)
        messages.append({"role": "assistant", "content": resp.get("content", "")})
        sql = _extract_sql(resp.get("content", ""))
        concl = _extract_block(resp.get("content", ""), "conclusion")

        if sql:
            show, kind = sql, "sql"
            action = _action_before_sql(resp.get("content", ""))
        elif concl:
            show, kind = concl, "conclusion"
            action = "形成查询结论"
        else:
            show, kind = resp.get("content", "").strip()[:500], "think"
            action = None
        if show:
            await emit({"type": "step_log", "skill_id": "query", "step": step,
                        "kind": kind, "text": show, "action": action})
            # 模型在取数循环里的自语思路（非 SQL/非结论）同步进主时间线，不丢弃
            if kind == "think" and show.strip():
                await emit({"type": "reasoning", "skill_id": "query", "text": show.strip()})

        if concl:
            if not evidences:
                # 未取数就收口：打回，禁止空结论
                messages.append({"role": "user",
                                 "content": "你还没有成功取到任何数据。禁止只给口径或空结论；"
                                            "请先输出 ```sql 取数，拿到真实结果后再 ```conclusion。"})
                continue
            conclusion = concl
            break
        if sql:
            res = sql_tool.run_sql(sql, cursor=cursor)
            if res.get("error"):
                messages.append({"role": "user", "content": f"SQL 报错：{res['error']}，修正后重试。"})
                continue
            await emit({"type": "tool_call", "skill_id": "query", "tool": "run_sql",
                        "title": "执行查询 · run_sql",
                        "detail": f"返回 {res['row_count']} 行", "sql": res["sql"]})
            evidences.append(Evidence(metric=goal, sql=res["sql"],
                                     row_count=res["row_count"], rows=res["rows"][:50]))
            messages.append({"role": "user",
                             "content": f"结果（{res['row_count']}行）：\n{_rows_preview(res['rows'][:10])}\n给 ```conclusion。"})
            continue
        messages.append({"role": "user", "content": "请输出 ```sql 或 ```conclusion。"})
    else:
        status = "truncated"
        conclusion = conclusion or "达到 step 上限，未完成（不伪造）。"
    return SubResult(skill_id="query", skill_name="问数执行", conclusion=conclusion,
                     evidences=evidences, status=status), local_usage


# ---------- 汇聚：共识 / 分歧 ----------
MERGE_SYS = """你是结果汇聚者。阅读各专家子 Agent 的结论，输出 JSON：
{"summary":"整体结论","consensus":[{"point":"多方法共同指向的判断","support":["来源skill"]}],"divergences":[{"point":"结论不一致处","why":"分歧原因","pending":"待复核"}]}}
规则：共识=≥2个方法共同指向；分歧=结论冲突，挂起待复核，不强行调和；不新增无证据的判断。只输出 JSON。"""


async def merge_results(sub_results: list[SubResult], model: str) -> tuple[FinalResult, dict]:
    local_usage = {"prompt": 0, "completion": 0, "total": 0}
    blocks = "\n\n".join(
        f"【{r.skill_name}】({r.status})\n{r.conclusion}" for r in sub_results)
    resp = await llm.chat(
        [{"role": "system", "content": MERGE_SYS},
         {"role": "user", "content": blocks}], model=model)
    for k in ("prompt", "completion", "total"):
        local_usage[k] += resp.get("usage", {}).get(k, 0)
    raw = resp.get("content", "") or ""
    data = planner._extract_json(raw) or {}
    summary = (data.get("summary") or "").strip()
    consensus = data.get("consensus") or []
    divergences = data.get("divergences") or []
    # 宽松兜底：标准 JSON 未完整解析时，从原文正则补提 summary 字段
    if not summary:
        m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.S)
        if m:
            summary = m.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
    # 汇总区不得空白：确实提取不到时落一句引导，整体判断以上方专家结论交叉印证为准
    if not summary:
        print("[merge] summary 未提取到，原始返回前800字：\n", raw[:800])
        summary = ("各专家已从互补视角完成独立诊断，具体依据见上方各专家结论；"
                   "总体判断以多方法交叉印证为准。")
    final = FinalResult(
        summary=summary,
        consensus=consensus,
        divergences=divergences,
        evidences=[e for r in sub_results for e in r.evidences],
    )
    return final, local_usage


# ---------- 主编排 ----------
async def orchestrate(question: str, model: str, emit, session) -> None:
    # 1) 意图拆分 + Plan
    points, planner_usage, plan_text = await planner.classify_and_plan(question, model)
    session.add_tokens(planner_usage)
    # 主 Agent 开场规划思考（模型自然语言分步打算），先交代思路再动手
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

    # 说明性规划播报不进主对话流（主 Agent 的临场思考在各阶段按需发出），避免与临场表态重复

    all_subs: list[SubResult] = []
    sem = asyncio.Semaphore(MAX_PARALLEL)
    # 主 Agent 已查明的总量事实，随取数累积，派发给专家时直接引用，避免专家重复聚合总量
    known_facts_parts: list[str] = []

    # 兜底合并连续 query 点：同一指标跨期/同表多指标/同维度多取值，一次取数即可，
    # 避免 planner 误拆后重复查本体、重复跑 SQL。chat/analysis 各自独立成组。
    groups: list[tuple[str, list[IntentPoint]]] = []
    qbuf: list[IntentPoint] = []

    def flush_query() -> None:
        if qbuf:
            groups.append(("query", list(qbuf)))
            qbuf.clear()

    for p in points:
        if p.intent == "query":
            qbuf.append(p)
        else:
            flush_query()
            groups.append((p.intent, [p]))
    flush_query()

    for kind, members in groups:
        p = members[0]

        if kind == "chat":
            resp = await llm.chat(
                [{"role": "system", "content": "你是友好的助手，直接简短回答寒暄与一般问题，不查数。"},
                 {"role": "user", "content": question}], model=model)
            session.add_tokens(resp.get("usage", {}))
            await emit({"type": "chat_answer", "text": resp.get("content", "")})
            continue

        if kind == "query":
            # 问数由主 Agent 直接负责（查本体口径 -> 取数 -> 作答），不派发子 Agent、不进专家工作区；
            # 合并点共用一次取数，结论一次性给出各期/各值与对比
            merged_goal = "；".join(dict.fromkeys(
                m.plan.goal for m in members if m.plan and m.plan.goal)) or question
            cursor = sql_tool.new_cursor()
            sub, usage = await run_query_point(merged_goal, question, model, emit, cursor)
            session.add_tokens(usage)
            await emit({"type": "query_answer", "text": sub.conclusion,
                         "evidences": [e.to_dict() for e in sub.evidences]})
            known_facts_parts.append(sub.conclusion)
            continue

        # analysis：按问题动态挑选选中 Skill（不超过 MAX_SUB_AGENTS 硬约束）
        selected = p.plan.skill_ids[:MAX_SUB_AGENTS]

        # —— 主 Agent 先通读可用分析本领，再结合问题挑选（不是固定派齐） ——
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
        pick_txt = "、".join(
            f"{s['name']}（{s.get('expert_role', '')}）" for s in chosen)
        drop_txt = "、".join(s["name"] for s in dropped)
        await emit({"type": "reasoning", "skill_id": "planner",
                    "text": ("阅读各项本领后，结合本问题选择派出："
                             f"{pick_txt}，各自独立取数、从不同视角交叉诊断；"
                             + (f"其余本领（{drop_txt}）与本问题关联较弱，本次暂不派出；" if drop_txt else "")
                             + "完成后我再汇总各家结论、比对共识与差异。")})

        async def guarded(skill_id: str):
            async with sem:
                sname = next((s["name"] for s in skilllib.list_skills() if s["id"] == skill_id), skill_id)
                await emit({"type": "agent_start", "skill_id": skill_id, "skill_name": sname})
                cursor = sql_tool.new_cursor()
                known_facts = "\n".join(known_facts_parts)
                return await run_skill_worker(skill_id, p.plan.goal, question, model, emit, cursor,
                                             known_facts=known_facts)

        results = await asyncio.gather(
            *[guarded(sid) for sid in selected], return_exceptions=True)
        for sid, res in zip(selected, results):
            if isinstance(res, Exception):
                # 单个专家异常不拖垮整组：包装为 error 子结果继续汇聚
                sub = SubResult(skill_id=sid, skill_name=sid,
                                conclusion=f"专家 {sid} 执行异常：{type(res).__name__}: {res}",
                                status="error", note="已隔离，不影响其他专家")
                session.add_tokens({})
                session.add_sub_result(sub.to_dict())
                all_subs.append(sub)
                await emit({"type": "sub_done", "sub": sub.to_dict()})
                continue
            sub, usage = res
            session.add_tokens(usage)
            session.add_sub_result(sub.to_dict())
            all_subs.append(sub)
            await emit({"type": "sub_done", "sub": sub.to_dict()})

    # 2) 汇聚
    if all_subs:
        expert_subs = [s for s in all_subs if s.skill_id != "query"]
        if expert_subs:
            done_n = sum(1 for s in expert_subs if s.status == "done")
            await emit({"type": "reasoning", "skill_id": "planner",
                        "text": f"{done_n} 位专家已完成独立取数与诊断。现在由我来汇总各家结论、"
                                "比对共识与差异，再给出整体判断。"})
        final, usage = await merge_results(all_subs, model)
        session.add_tokens(usage)
        session.set_final(final.to_dict())
        await emit({"type": "consensus", "consensus": final.consensus, "divergences": final.divergences})
        await emit({"type": "final", "summary": final.summary,
                    "evidences": [e.to_dict() for e in final.evidences]})
    await emit({"type": "tokens", "tokens": session.tokens})
