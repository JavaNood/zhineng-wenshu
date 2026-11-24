"""worker：子 Agent，执行单个 Skill，step≤60，SQL 取数循环，产出结论+证据。"""
from __future__ import annotations
import re
from .. import llm
from ..tools import sql_tool, skill_tools
from ..skills import loader as skilllib
from ..ontology import loader as onto
from ..schemas import SubResult, Evidence
from ..config import MAX_STEPS


WORKER_SYS = """你是一位被派遣的分析专家子 Agent，严格按给定 Skill 的操作指引取数分析。
【工作方式】
1. 先看下方"数据表与字段"确定表别名、列名与允许 join，禁止用 information_schema 盲探表结构。
2. 每一步只做一件事：需要取数时输出一条 DuckDB 只读 SQL（```sql 代码块），表用英文别名，列用中文原名；输出 SQL 前先用一句完整、明确的中文写明本步动作，格式"步骤N：<明确的查询对象 + 口径 + 时间/范围>"，例如"步骤一：分别统计2024与2025全年销售额并计算同比变化"。必须写清查什么、按什么口径、看哪个范围；禁止半句、禁止未闭合括号的残句、禁止只写"订单量）"这类碎片。
3. 拿到 SQL 结果后继续下一步；数据不足可修正 SQL 重试（在 step 上限内）。
4. 全部取数完成后，输出最终结论 ```conclusion 代码块：先用数据给出支撑判断，末尾以"## 该专家的主要结论"作为小标题，给出本视角下的一句话核心结论。
【硬约束】
- 未成功执行过任何 SQL、没有真实结果前，禁止输出 ```conclusion，禁止只复述口径或空泛判断。
- 只用下方本体给定的标准指标口径，禁止自造指标或编造数字，结论必须基于真实 SQL 结果。
- 时间过滤：orders.订单日期 可直接按年/季过滤；需要促销标记/标准年月时再 join d_date。
数据表与数据地图：{data_map}"""


def _extract_block(text: str, tag: str) -> str | None:
    m = re.search(rf"```{tag}\s*([\s\S]*?)```", text or "")
    return m.group(1).strip() if m else None


def _extract_sql(text: str) -> str | None:
    m = re.search(r"```sql\s*([\s\S]*?)```", text or "")
    return m.group(1).strip() if m else None


async def run_skill_worker(
    skill_id: str,
    goal: str,
    question: str,
    model: str,
    emit,
    cursor,
    known_facts: str = "",
) -> SubResult:
    card = skilllib.get_skill(skill_id)
    local_usage = {"prompt": 0, "completion": 0, "total": 0}
    if not card:
        return SubResult(skill_id=skill_id, skill_name=skill_id,
                         conclusion=f"未找到 Skill {skill_id}", status="error"), local_usage

    skill_name = card.get("title", card["name"])
    evidences: list[Evidence] = []
    ontology_brief = "\n".join(
        f"- {m['name']}({m['code']} = {m['expression'] or m['formula']})"
        for m in onto.list_metrics())

    # —— 显式查阅本体（体现语义层作用）：先对齐指标口径，再确认表结构与关联 ——
    hit = onto.resolve_metric(goal) or onto.resolve_metric(question)
    if hit:
        expr = hit.get("expression") or hit.get("formula")
        dims = hit.get("dims") or []
        dims_txt = "、".join(dims) if dims else "未限定"
        body = (f"指标：{hit['name']}（{hit['code']}）\n"
                f"口径：{expr}\n单位：{hit.get('unit', '')}\n"
                f"可下钻维度：{dims_txt}")
        await emit({"type": "tool_call", "skill_id": skill_id, "tool": "resolve_metric",
                    "title": "查阅本体 · 对齐指标口径",
                    "detail": f"{hit['name']}（{hit['code']}）= {expr}；单位 {hit.get('unit', '')}",
                    "body": body})
    ds = onto.load_ontology().get("datasets", [])
    rels = onto.get_relationships()
    rel_lines = [f"{r['from']} → {r['to']}" for r in rels]
    body_ont = ("数据表：\n" + "\n".join(f"- {d['source']}" for d in ds)
                + "\n允许 join 白名单：\n" + "\n".join(f"- {r}" for r in rel_lines))
    await emit({"type": "tool_call", "skill_id": skill_id, "tool": "read_ontology",
                "title": "查阅本体 · 表结构与允许关联",
                "detail": f"已读取 {len(ds)} 张数据表与 join 白名单，仅允许白名单内关联",
                "body": body_ont})

    # 主 Agent 派发给本专家的任务说明（前端可点击查看提示词）
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

    conclusion = ""
    status = "done"
    for step in range(1, MAX_STEPS + 1):
        await emit({"type": "agent_progress", "skill_id": skill_id,
                    "skill_name": skill_name, "step": step})

        async def _reasoning(t: str):
            await emit({"type": "reasoning", "skill_id": skill_id, "text": t})

        resp = await llm.chat_stream(messages, model=model, on_reasoning=_reasoning)
        for k in ("prompt", "completion", "total"):
            local_usage[k] += resp.get("usage", {}).get(k, 0)
        messages.append({"role": "assistant", "content": resp.get("content", "")})

        sql = _extract_sql(resp.get("content", ""))
        concl = _extract_block(resp.get("content", ""), "conclusion")

        # 结构化步骤时间线：按专家记录每一步（SQL / 思考 / 收口），供前端分步展示
        if sql:
            show, kind = sql, "sql"
            action = _action_before_sql(resp.get("content", ""), sql)
        elif concl:
            show, kind = concl, "conclusion"
            action = "形成分析结论"
        else:
            show, kind = resp.get("content", "").strip()[:500], "think"
            action = None
        if show:
            await emit({"type": "step_log", "skill_id": skill_id, "step": step,
                        "kind": kind, "text": show, "action": action})

        if concl:
            if not evidences:
                messages.append({"role": "user",
                                 "content": "你还没有成功取到任何数据。禁止只给口径或空泛判断；"
                                            "请先输出 ```sql 取数，拿到真实结果后再 ```conclusion。"})
                continue
            conclusion = concl
            break
        if sql:
            res = sql_tool.run_sql(sql, cursor=cursor)
            if res.get("error"):
                messages.append({"role": "user",
                                 "content": f"SQL 执行报错：{res['error']}。请修正后重试，不要编造结果。"})
                continue
            # 记录证据
            ev = Evidence(metric=goal, sql=res["sql"], row_count=res["row_count"],
                          rows=res["rows"][:50])
            evidences.append(ev)
            await emit({"type": "tool_call", "skill_id": skill_id, "tool": "run_sql",
                        "title": "执行查询 · run_sql",
                        "detail": f"返回 {res['row_count']} 行", "sql": res["sql"]})
            preview = _rows_preview(res["rows"][:10])
            messages.append({"role": "user",
                             "content": f"SQL 结果（{res['row_count']} 行）：\n{preview}\n继续下一步，或给出 ```conclusion。"})
            continue
        # 既无 SQL 也无 conclusion：把模型输出当作试探，要求明确动作
        messages.append({"role": "user",
                         "content": "请输出 ```sql 取数，或在完成后输出 ```conclusion。"})
    else:
        status = "truncated"
        conclusion = conclusion or "达到 step 上限，结论不完整（已中止，不伪造完成）。"

    return SubResult(skill_id=skill_id, skill_name=skill_name,
                     conclusion=conclusion, evidences=evidences, status=status), local_usage


def _action_before_sql(text: str, sql: str | None = None) -> str | None:
    """提取 SQL 代码块之前的一句中文动作说明，作为右侧步骤标题。"""
    if not text:
        return None
    pre = text.split("```sql")[0] if "```sql" in text else text
    lines = [ln.strip(" ：:；;·-") for ln in pre.splitlines() if ln.strip()]
    for ln in reversed(lines):
        cn = re.findall(r"[一-鿿][一-鿿，、：:？?！!（）()0-9A-Za-z.%\-]+", ln)
        if cn:
            s = cn[-1].strip()
            if len(s) >= 4:
                return s[:60]
    return None


def _rows_preview(rows: list[dict]) -> str:
    if not rows:
        return "(空结果)"
    cols = list(rows[0].keys())
    out = [" | ".join(cols)]
    for r in rows:
        out.append(" | ".join(str(r.get(c, "")) for c in cols))
    return "\n".join(out)
