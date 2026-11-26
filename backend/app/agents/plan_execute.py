"""子 Agent：先由模型写出计划，再按计划逐步取数。思维链挂在对应步骤上，不拆成步骤标题。"""
from __future__ import annotations
import re

from .. import llm
from ..config import MAX_STEPS, PLAN_STEPS, RESULT_ROWS
from ..ontology import loader as onto
from ..schemas import Evidence, SubResult
from ..tools import skill_tools, sql_tool
from .react_graph import WORKER_SYS
from .textutil import _extract_block, _extract_json, _extract_sql, _rows_preview


_PLAN_ASK = """先制定计划，不要写 SQL，不要写结论。
只输出：
```plan
{"steps":[{"n":1,"action":"一句完整中文，写清查什么、口径、时间或范围"}]}
```
需要几步就写几步，最多 """ + str(PLAN_STEPS) + """ 步。每步只做一件取数。已知事实里已经有的总量不要再聚合，把步骤用在本技能的下钻。"""

_TRUNCATED = "达到 step 上限，结论不完整（已中止，不伪造完成）。"


def _parse_plan(content: str) -> list[dict]:
    matched = re.search(r"```plan\s*([\s\S]*?)```", content or "", re.I)
    data = _extract_json(matched.group(1) if matched else (content or ""))
    raw = data.get("steps") if isinstance(data, dict) else None
    steps: list[dict] = []
    if not isinstance(raw, list):
        return steps
    for item in raw:
        if isinstance(item, str):
            action = item.strip()
        elif isinstance(item, dict):
            action = str(item.get("action") or item.get("text") or "").strip()
        else:
            action = ""
        if len(action) < 4:
            continue
        steps.append({"n": len(steps) + 1, "action": action})
        if len(steps) >= PLAN_STEPS:
            break
    return steps


def _exec_ask(step: dict) -> str:
    return (
        f"现在只执行计划的第 {step['n']} 步：{step['action']}\n"
        "只输出一条 ```sql。不要写结论，不要执行别的步骤。"
    )


def _ontology_brief() -> str:
    return "\n".join(
        f"- {m['name']}({m['code']} = {m['expression'] or m['formula']})"
        for m in onto.list_metrics()
    )


async def run_plan_execute(
    *,
    skill_id: str,
    skill_name: str,
    goal: str,
    question: str,
    model: str,
    emit,
    cursor,
    known_facts: str,
    card: dict,
) -> tuple[SubResult, dict]:
    skill_card = skill_tools.render_skill_card(card)
    known_block = (
        f"【主 Agent 已查明的总量事实，可直接引用，禁止重复聚合同一总量】\n{known_facts}\n"
        if known_facts else ""
    )
    context = "\n".join([
        f"原始问题：{question}",
        f"分析目标：{goal}",
        f"【本体标准指标口径，必须遵守，禁止自造】\n{_ontology_brief()}",
        skill_card,
        known_block.strip(),
    ]).strip()
    prompt_text = (
        f"【原始问题】{question}\n【分析目标】{goal}\n"
        f"【你的专家方法（Skill）】\n{skill_card}\n"
        + (f"\n{known_block}" if known_block else "")
        + "【任务要求】先给出本视角的取数计划，再按计划逐步执行；只基于真实 SQL 结果下结论。"
    )
    await emit({"type": "agent_prompt", "skill_id": skill_id, "prompt": prompt_text})

    messages = [
        {"role": "system", "content": WORKER_SYS},
        {"role": "user", "content": context + "\n\n" + _PLAN_ASK},
    ]
    usage = {"prompt": 0, "completion": 0, "total": 0}
    plan: list[dict] = []
    phase = "plan"
    index = 0
    evidences: list[Evidence] = []
    conclusion = ""

    for step in range(1, MAX_STEPS + 1):
        await emit({
            "type": "agent_progress", "skill_id": skill_id,
            "skill_name": skill_name, "step": step,
        })
        held: list[str] = []

        async def _hold(text: str, box: list[str] = held) -> None:
            box.append(text)

        resp = await llm.chat_stream(messages, model=model, on_reasoning=_hold)
        for k in usage:
            usage[k] += (resp.get("usage") or {}).get(k, 0)
        content = resp.get("content") or ""
        thought = "".join(held).strip()
        messages.append({"role": "assistant", "content": content})

        if phase == "plan":
            parsed = _parse_plan(content)
            if not parsed:
                messages.append({
                    "role": "user",
                    "content": "没有解析到 ```plan JSON。请只输出计划，不要写 SQL。",
                })
                continue
            plan = parsed
            await emit({
                "type": "worker_plan",
                "skill_id": skill_id,
                "thought": thought,
                "steps": plan,
            })
            phase = "exec"
            index = 0
            messages.append({"role": "user", "content": _exec_ask(plan[0])})
            continue

        if phase == "exec":
            sql = _extract_sql(content)
            if not sql:
                messages.append({
                    "role": "user",
                    "content": "这一步需要一条 ```sql，对应计划里的这一句，不要写结论。",
                })
                continue
            res = sql_tool.run_sql(sql, cursor=cursor)
            if res.get("error"):
                messages.append({
                    "role": "user",
                    "content": f"SQL 执行报错：{res['error']}。请只修正这一步的 SQL，不要编造结果。",
                })
                continue
            action = plan[index]["action"]
            ev = Evidence(metric=goal, sql=res["sql"], row_count=res["row_count"], rows=res["rows"][:RESULT_ROWS])
            evidences.append(ev)
            await emit({
                "type": "step_log",
                "skill_id": skill_id,
                "step": plan[index]["n"],
                "kind": "sql",
                "text": res["sql"],
                "action": action,
                "thought": thought,
            })
            await emit({
                "type": "tool_call",
                "skill_id": skill_id,
                "tool": "run_sql",
                "title": action,
                "detail": f"返回 {res['row_count']} 行",
                "sql": res["sql"],
            })
            preview = _rows_preview(res["rows"][:RESULT_ROWS], res["row_count"])
            index += 1
            if index < len(plan):
                messages.append({
                    "role": "user",
                    "content": (
                        f"SQL 结果（{res['row_count']} 行）：\n{preview}\n"
                        + _exec_ask(plan[index])
                    ),
                })
                continue
            phase = "conclude"
            messages.append({
                "role": "user",
                "content": (
                    f"SQL 结果（{res['row_count']} 行）：\n{preview}\n"
                    "计划中的取数都已完成。根据这些真实结果输出 ```conclusion。"
                    "末尾用「## 该专家的主要结论」给出一句话。"
                ),
            })
            continue

        concl = _extract_block(content, "conclusion")
        if concl and evidences:
            conclusion = concl
            break
        if not evidences:
            messages.append({
                "role": "user",
                "content": "还没有成功的 SQL 结果。请回到计划，先输出 ```sql。",
            })
            phase = "exec" if plan else "plan"
            continue
        messages.append({
            "role": "user",
            "content": "请把结论放进 ```conclusion，并且只能使用上面的 SQL 结果。",
        })

    else:
        conclusion = conclusion or _TRUNCATED
        status = "truncated"
        sub = SubResult(
            skill_id=skill_id, skill_name=skill_name, conclusion=conclusion,
            evidences=evidences, status=status,
        )
        return sub, usage

    sub = SubResult(
        skill_id=skill_id, skill_name=skill_name, conclusion=conclusion,
        evidences=evidences, status="done",
    )
    return sub, usage
