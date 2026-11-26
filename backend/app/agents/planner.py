"""planner：意图三分类拆分、Plan 生成、分析点动态选 Skill（≥2）。"""
from __future__ import annotations
from .. import llm
from ..semantic import loader as sem
from ..skills import loader as skilllib
from ..schemas import IntentPoint, Plan, PlanStep
from .textutil import _extract_json


def _ontology_brief() -> str:
    metrics = "、".join(f"{m['name']}({m['code']})" for m in sem.list_metrics())
    return f"标准指标：{metrics}"


def _skill_brief() -> str:
    return "\n".join(f"- {s['id']}｜{s['name']}：{s['trigger']}" for s in skilllib.list_skills())


PLANNER_SYS = """你是智能问数系统的规划器。把用户问题拆成若干独立问题点，每点归类为：
- chat：寒暄/确认/与数据无关的一般问题，直接回答，不查数。
- query：查询某个指标/数值/事实（如"2025年销售额""环比如何"）。
- analysis：解释变动原因、诊断异常、对比评估（如"销售额为什么下降"）。

规则：
1. 只在"需要不同取数动作/不同回答方式"时才拆点。**同一次查询就能拿到的内容，必须合并成一个 query 点，禁止拆开重复取数**：
   - 同一指标跨多个时间（2025与2024销售额、Q1/Q2/Q3/Q4各季、本月与上月）→ 合并为一个点，goal 写"分别给出各期数值并计算对比"。
   - 同一表可一次聚合的多个指标（销售额、订单量、客单价）→ 合并为一个点。
   - 同一维度下的多个取值（各渠道销售额、各大区销售额、Top门店）→ 合并为一个点，goal 写"按该维度分组列出"。
2. analysis 点必须从给定 Skill 列表里选 **不少于2个** 互补的分析方法（skills），不写死，按问题匹配。
3. query 点给出取数目标与涉及的标准指标；analysis 点给出总量确认目标与所选 skills。
4. 只能使用给定标准指标口径，禁止自造指标。
5. 判定基准：能否用一条 SQL（含 GROUP BY / 多期对比）一次取回？能就合并，不能才拆。

先在 plan 字段用自然语言写出主 Agent 的分步执行打算（先做什么、再做什么、最后做什么，像给用户交代思路，不超过3句），再输出结构化 points。

只输出 JSON，不要多余解释，格式：
{"plan":"主Agent的分步执行打算","points":[{"index":0,"text":"原问句片段","intent":"query|analysis|chat","goal":"该点目标","skills":["skill_id"],"steps":[{"step_id":"s1","action":"动作","query_spec":"取数说明"}]}]}
analysis 点的 skills 填选中的 skill_id；query/chat 点 skills 为空数组。"""


def _usage_add(a: dict, b: dict) -> dict:
    out = {"prompt": 0, "completion": 0, "total": 0}
    for src in (a or {}, b or {}):
        for k in out:
            out[k] += src.get(k, 0) or 0
    return out


def _points_from(data, question: str, *, fallback: bool) -> tuple[list[IntentPoint], str]:
    plan_text = (data.get("plan") or "").strip() if isinstance(data, dict) else ""
    raw_points = data.get("points", []) if isinstance(data, dict) else []
    points: list[IntentPoint] = []
    for i, p in enumerate(raw_points):
        intent = p.get("intent", "chat")
        if intent not in ("chat", "query", "analysis"):
            intent = "chat"
        steps = [PlanStep(step_id=s.get("step_id", f"s{j+1}"),
                          action=s.get("action", ""),
                          query_spec=s.get("query_spec", ""))
                 for j, s in enumerate(p.get("steps", []))]
        skill_ids = []
        if intent == "analysis":
            skill_ids = [s for s in (p.get("skills") or []) if skilllib.get_skill(s)]
        plan = Plan(intent=intent, goal=p.get("goal", ""), steps=steps, skill_ids=skill_ids)
        points.append(IntentPoint(index=i, raw_text=p.get("text", ""),
                                  intent=intent, plan=plan))
    if not points and fallback:
        points.append(IntentPoint(index=0, raw_text=question, intent="query",
                                  plan=Plan(intent="query", goal=question)))
    return points, plan_text


def _analysis_short(points: list[IntentPoint]) -> bool:
    return any(
        p.intent == "analysis" and p.plan is not None and len(p.plan.skill_ids) < 2
        for p in points
    )


async def classify_and_plan(
    question: str, model: str, on_reasoning=None,
) -> tuple[list[IntentPoint], dict, str]:
    user = (
        f"【本体】\n{_ontology_brief()}\n\n"
        f"【可用 Skill（专家方法）】\n{_skill_brief()}\n\n"
        f"【用户问题】{question}"
    )
    messages = [
        {"role": "system", "content": PLANNER_SYS},
        {"role": "user", "content": user},
    ]
    resp = await llm.chat(messages, model=model, on_reasoning=on_reasoning)
    usage = resp.get("usage", {})
    content = resp.get("content", "")
    points, plan_text = _points_from(_extract_json(content), question, fallback=True)
    # 技能选少了就再问一次模型，不用技能列表的前两项冒充它的选择。
    if _analysis_short(points):
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": (
                "analysis 点的 skills 少于 2 个，或含有不在技能列表中的 id。"
                "请只根据用户问题和技能列表重新输出完整 JSON："
                "每个 analysis 点从列表中选择不少于 2 个真实存在的 skill_id，"
                "plan 仍用自然语言说明打算。不要解释，只输出 JSON。"
            )},
        ]
        resp2 = await llm.chat(messages, model=model, on_reasoning=on_reasoning)
        usage = _usage_add(usage, resp2.get("usage", {}))
        data2 = _extract_json(resp2.get("content", ""))
        if isinstance(data2, dict) and data2.get("points"):
            points2, plan_text2 = _points_from(data2, question, fallback=False)
            if points2:
                points = points2
                if plan_text2:
                    plan_text = plan_text2
    return points, usage, plan_text
