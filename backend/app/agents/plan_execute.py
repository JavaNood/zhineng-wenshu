"""子 Agent：先计划，执行阶段用与主 Agent 相同的查数工具做 ReAct，并可修订未执行步骤。"""
from __future__ import annotations
import re

from .. import llm
from ..config import MAX_STEPS, PLAN_STEPS, RESULT_ROWS
from ..schemas import Evidence, SubResult
from ..tools import sql_tool
from ..tools.batch import classify, formula_of, label, plan_hit, run_readonly, schema_hit
from ..tools.schemas import assistant_message, calls_to_pending, tool_message, worker_tools
from .react_graph import worker_sys
from .textutil import _extract_block, _extract_json, _rows_preview


_PLAN_ASK = """先制定计划，不要写 SQL，不要调用工具，不要写结论。
只输出：
```plan
{"steps":[{"n":1,"action":"一句完整中文，写清查什么、口径、时间或范围"}]}
```
需要几步就写几步，最多 """ + str(PLAN_STEPS) + """ 步。每步只写一条即将执行的取数：指标、维度、时间或范围。不要把 resolve_certified_metric、read_sql_plan、describe_schema、read_entity 写成计划步骤。已知事实里已经有的总量不要再聚合，把步骤用在本技能的下钻。要放弃还没执行的步骤，单独输出一个 steps 为空数组的 ```plan。"""

_TRUNCATED = "达到 step 上限，结论不完整（已中止，不伪造完成）。"


def _plan_steps(content: str) -> list[dict] | None:
    """解析 ```plan。返回 None 表示没有合法计划；空列表表示放弃剩余步骤。"""
    matched = re.search(r"```plan\s*([\s\S]*?)```", content or "", re.I)
    data = _extract_json(matched.group(1) if matched else (content or ""))
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    raw = data["steps"]
    steps: list[dict] = []
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


def _parse_plan(content: str) -> list[dict]:
    return _plan_steps(content) or []


def _renumber(steps: list[dict]) -> list[dict]:
    return [{"n": i + 1, "action": s["action"]} for i, s in enumerate(steps[:PLAN_STEPS])]


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
    ontology_enabled: bool = True,
) -> tuple[SubResult, dict]:
    from ..tools import skill_tools
    skill_card = skill_tools.render_skill_card(card)
    known_block = (
        f"【主 Agent 已查明的总量事实，可直接引用，禁止重复聚合同一总量】\n{known_facts}\n"
        if known_facts else ""
    )
    prompt_text = (
        f"【原始问题】{question}\n【分析目标】{goal}\n"
        f"【你的专家方法（Skill）】\n{skill_card}\n"
        + (f"\n{known_block}" if known_block else "")
        + "【任务要求】先给出本视角的取数计划，再按计划调用工具并执行；只基于真实 SQL 结果下结论。"
    )
    await emit({"type": "agent_prompt", "skill_id": skill_id, "prompt": prompt_text})
    opening = "\n".join([
        f"原始问题：{question}",
        f"分析目标：{goal}",
        skill_card,
        known_block.strip(),
        _PLAN_ASK,
    ]).strip()
    messages = [
        {"role": "system", "content": worker_sys(ontology_enabled)},
        {"role": "user", "content": opening},
    ]
    usage = {"prompt": 0, "completion": 0, "total": 0}
    plan: list[dict] = []
    phase = "plan"
    index = 0
    evidences: list[Evidence] = []
    conclusion = ""
    metric_ready = False
    plan_ready = False
    schema_ready = False
    formula = ""

    for step in range(1, MAX_STEPS + 1):
        await emit({
            "type": "agent_progress", "skill_id": skill_id,
            "skill_name": skill_name, "step": step,
        })
        offered = None if phase == "plan" else worker_tools(ontology_enabled)
        resp = await llm.chat_stream(messages, model=model, tools=offered)
        for k in usage:
            usage[k] += (resp.get("usage") or {}).get(k, 0)
        content = resp.get("content") or ""
        tool_calls = list(resp.get("tool_calls") or [])
        messages.append(assistant_message(content, resp.get("reasoning") or "", tool_calls))

        if phase == "plan":
            parsed = _plan_steps(content)
            if not parsed:
                messages.append({"role": "user", "content": "没有解析到 ```plan JSON。请只输出计划，不要写 SQL，不要调用工具。"})
                continue
            plan = parsed
            await emit({"type": "worker_plan", "skill_id": skill_id, "thought": "", "steps": plan})
            phase = "exec"
            index = 0
            messages.append({
                "role": "user",
                "content": "计划已记录。现在用函数执行。一次回复最多 4 个函数。写 SQL 前先调用函数取得这一步需要的口径和规划。不要在正文里写工具参数。",
            })
            continue

        has_plan = "```plan" in content.lower()
        tools = calls_to_pending(tool_calls)
        if tools and has_plan:
            _metric, plan_ready, schema_ready, formula, index, pieces = await _run_tools(
                tools, skill_id, emit, cursor, goal, plan, index, evidences,
                ontology_enabled, metric_ready, plan_ready, schema_ready, formula,
            )
            metric_ready = _metric
            _append_tool_results(
                messages, pieces,
                "同一次回复里既有工具又有新计划。这一轮只处理工具，计划不改。请在看见结果后再单独修订计划。",
            )
            continue
        if tools:
            metric_ready, plan_ready, schema_ready, formula, index, pieces = await _run_tools(
                tools, skill_id, emit, cursor, goal, plan, index, evidences,
                ontology_enabled, metric_ready, plan_ready, schema_ready, formula,
            )
            extra = ""
            if evidences and index >= len(plan):
                extra = "计划中的取数都已完成。根据这些真实结果输出 ```conclusion。末尾用「## 该专家的主要结论」给出一句话。"
            _append_tool_results(messages, pieces, extra)
            continue
        if has_plan:
            fresh = _plan_steps(content)
            if fresh is None:
                messages.append({"role": "user", "content": "没有解析到新的 ```plan。未执行步骤保持不变。"})
                continue
            done = plan[:index]
            plan = _renumber(done + fresh)
            await emit({"type": "worker_plan", "skill_id": skill_id, "thought": "", "steps": plan})
            pending = "；".join(s["action"] for s in plan[index:]) or "（没有未执行步骤）"
            messages.append({"role": "user", "content": "计划已修订。尚未执行的步骤：" + pending})
            continue
        concl = _extract_block(content, "conclusion")
        if concl and evidences:
            if index < len(plan):
                pending = "；".join(s["action"] for s in plan[index:])
                messages.append({
                    "role": "user",
                    "content": "还有未执行步骤：" + pending + "。先完成这些取数，或单独用 ```plan {\"steps\":[]} 放弃剩余步骤，然后再写 ```conclusion。",
                })
                continue
            conclusion = concl
            break
        messages.append({
            "role": "user",
            "content": "这一步需要调用函数，或在看见结果后单独用 ```plan 修订未执行步骤，或在已有 SQL 结果后输出 ```conclusion。一次最多 4 个函数。不要在正文里写工具参数。",
        })

    else:
        conclusion = conclusion or _TRUNCATED
        return SubResult(
            skill_id=skill_id, skill_name=skill_name, conclusion=conclusion,
            evidences=evidences, status="truncated",
        ), usage

    return SubResult(
        skill_id=skill_id, skill_name=skill_name, conclusion=conclusion,
        evidences=evidences, status="done",
    ), usage


def _append_tool_results(messages: list[dict], pieces: list[tuple[str, str]], note: str) -> None:
    if not pieces:
        if note:
            messages.append({"role": "user", "content": note})
        return
    if note:
        call_id, text = pieces[-1]
        pieces[-1] = (call_id, text + "\n" + note)
    for cid, body in pieces:
        messages.append(tool_message(cid, body))


async def _run_tools(tools, skill_id, emit, cursor, goal, plan, index, evidences,
                     ontology_enabled, metric_ready, plan_ready, schema_ready, formula):
    decisions = classify(
        tools, ontology_enabled=ontology_enabled, metric_ready=metric_ready,
        plan_ready=plan_ready, schema_ready=schema_ready, skills_listed=False,
        has_sql=bool(evidences), dispatched=False, allow_orchestrator=False,
    )
    pieces: list[tuple[str, str]] = []
    any_skipped = False
    for item in decisions:
        tool = item["tool"]
        name = str(tool.get("tool") or "")
        call_id = str(tool.get("call_id") or "")
        if not item["run"]:
            any_skipped = True
            pieces.append((call_id, "未执行：" + item["reason"]))
            await emit({
                "type": "tool_call", "skill_id": skill_id, "tool": name or "unknown",
                "title": "未执行", "detail": item["reason"], "body": label(tool),
            })
            continue
        if name == "run_sql":
            sql = str(tool.get("sql") or "")
            res = sql_tool.run_sql(sql, cursor=cursor)
            action = plan[index]["action"] if index < len(plan) else "run_sql"
            if res.get("error"):
                body = res["error"]
                detail = body
                title = "run_sql"
            else:
                shown = res["rows"][:RESULT_ROWS]
                body = f"SQL 结果（{res['row_count']} 行）：\n{_rows_preview(shown, res['row_count'])}"
                detail = f"返回 {res['row_count']} 行"
                title = action
                ev = Evidence(metric=goal, formula=formula, sql=res["sql"], row_count=res["row_count"], rows=shown)
                evidences.append(ev)
                await emit({
                    "type": "step_log", "skill_id": skill_id, "step": index + 1,
                    "kind": "sql", "text": res["sql"], "action": action, "thought": "",
                })
                index += 1
            pieces.append((call_id, body))
            await emit({
                "type": "tool_call", "skill_id": skill_id, "tool": "run_sql",
                "title": title, "detail": detail, "sql": res.get("sql") or sql,
            })
            continue
        body = run_readonly(tool)
        pieces.append((call_id, body))
        await emit({
            "type": "tool_call", "skill_id": skill_id, "tool": name,
            "title": name, "detail": label(tool)[:180], "body": body,
        })
        if name == "resolve_certified_metric":
            got = formula_of(body)
            if got:
                metric_ready = True
                formula = got
        elif name == "read_sql_plan" and plan_hit(body):
            plan_ready = True
        elif name == "describe_schema" and schema_hit(body):
            schema_ready = True
    if pieces and any_skipped:
        call_id, text = pieces[-1]
        pieces[-1] = (call_id, text + "\n请在下一次回复里继续调用未执行的这些工具。")
    return metric_ready, plan_ready, schema_ready, formula, index, pieces
