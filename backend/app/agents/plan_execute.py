"""子 Agent：先计划，执行阶段用与主 Agent 相同的查数工具做 ReAct，并可修订未执行步骤。"""
from __future__ import annotations
import re

from .. import llm
from ..config import MAX_STEPS, PLAN_SOFT, PLAN_STEPS, RESULT_ROWS, SQL_IDLE_NUDGE, SQL_IDLE_STOP
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
计划控制在 """ + str(PLAN_SOFT) + """ 步以内，最多 """ + str(PLAN_STEPS) + """ 步。超过 """ + str(PLAN_STEPS) + """ 步的整份计划不会收下。每步只写一条即将执行的取数：指标、维度、时间或范围。不要把 resolve_certified_metric、read_sql_plan、describe_schema、read_entity 写成计划步骤。计划只保留这个问题要回答的取数，技能里和本题无关的下钻不要写进去。指标用问题点名的那个，技能里的销售额、毛利额只是例子。已知事实里已经有的总量不要再聚合。执行中可以再交 ```plan，新增、改写或删除尚未执行的步骤，总数仍不超过 """ + str(PLAN_STEPS) + """ 步。做完 """ + str(PLAN_SOFT) + """ 步后建议收拢；已经写在计划里的可以做完，也可以再追加到 """ + str(PLAN_STEPS) + """ 步。第 """ + str(PLAN_STEPS) + """ 步完成后只写结论。要放弃还没执行的步骤，单独输出一个 steps 为空数组的 ```plan。"""

_TRUNCATED = "达到 step 上限，结论不完整（已中止，不伪造完成）。"
_STALLED = "连续多轮没有新的查询结果，已停止该专家，不伪造结论。"
_GOAL_SCOPE = (
    "这段分析目标是全体专家共用的问题，只说明指标、时间和要解释的变化，不是你的任务清单。"
    "计划只写你的技能方法里、并且这道题需要的取数。技能里没有的角度不要写进计划，也不要写进结论。"
)


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
    return steps


def _parse_plan(content: str) -> list[dict]:
    return _plan_steps(content) or []


def _renumber(steps: list[dict]) -> list[dict]:
    return [{"n": i + 1, "action": s["action"]} for i, s in enumerate(steps)]


def _norm_action(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _join_actions(steps: list[dict]) -> str:
    return "；".join(s["action"] for s in steps) or "（没有未执行步骤）"


_SOFT = (
    f"已完成 {PLAN_SOFT} 步计划。建议收拢结论。"
    f"已经写在计划里的步骤可以做完，还可以追加步骤，总数最多 {PLAN_STEPS} 步。"
    "数已经够回答时直接写 ```conclusion。"
    "先写支撑判断，末尾用「## 该专家的主要结论」给出判断。"
)
_EXTEND = (
    "当前计划中的取数都已完成。请根据这些真实结果写 ```conclusion。"
    "先写支撑判断，末尾用「## 该专家的主要结论」给出判断。"
    f"只有还缺一块数据时，才另起一轮用 ```plan 追加步骤，总数最多 {PLAN_STEPS} 步。"
)
_LOCK = (
    f"计划已到 {PLAN_STEPS} 步上限。请基于当前查询结果写 ```conclusion。"
    "有结果支撑的判断写上，没有结果的点不写。不再接受新的计划和查询。"
    "先写支撑判断，末尾用「## 该专家的主要结论」给出判断。"
    "不要写查询，也不要写「分析已完成」这类执行状态。"
)
_OVERFLOW = (
    "未执行：当前没有未执行的计划步骤。请直接写 ```conclusion。"
    f"只有还缺数据时，才另起一轮用 ```plan 追加步骤，总数最多 {PLAN_STEPS} 步。"
)
_TOOLISH = re.compile(r"DSML|invoke\s+name\s*=", re.I)


def _toolish(text: str) -> bool:
    body = (text or "").strip()
    if not body:
        return False
    if _TOOLISH.search(body):
        return True
    compact = re.sub(r"\s+", " ", body)
    return bool(re.match(r"(SELECT|WITH)\b", compact, re.I))


def _take_conclusion(content: str, allow_prose: bool) -> str:
    """只收下判断。工具标记和一条裸查询不是结论。计划做完后，同一轮里的结论优先于计划和查询。"""
    fenced = _extract_block(content, "conclusion")
    if fenced and not _toolish(fenced):
        return fenced.strip()
    if not allow_prose:
        return ""
    body = re.sub(r"```plan[\s\S]*?```", " ", content or "", flags=re.I)
    body = re.sub(r"```sql[\s\S]*?```", " ", body, flags=re.I).strip()
    if len(body) >= 40 and not _toolish(body):
        return body
    return ""


def _ready_conclusion(content: str, evidences: list, plan: list, index: int) -> str:
    if not evidences:
        return ""
    plan_done = bool(plan) and (index >= len(plan) or index >= PLAN_STEPS)
    return _take_conclusion(content, plan_done)


def _tail_done_note(index: int) -> str:
    if index >= PLAN_STEPS:
        return _LOCK
    if index >= PLAN_SOFT:
        return _SOFT + "\n" + _EXTEND
    return _EXTEND


def _batch_note(plan: list[dict], index: int, before: int, evidences: list, follow: str) -> str:
    parts = [follow] if follow else []
    if evidences and index >= PLAN_STEPS:
        parts.append(_LOCK)
    else:
        if before < PLAN_SOFT <= index:
            parts.append(_SOFT)
        if evidences and plan and index >= len(plan):
            parts.append(_EXTEND)
    return "\n".join(parts)


def _exec_note(plan, index, before, ontology_enabled, metric_ready, plan_ready, schema_ready, evidences) -> str:
    """口径已经齐、这一轮又没有新 SQL 时，告诉模型下一回复直接查当前步骤。"""
    if evidences and (index >= len(plan) or index >= PLAN_STEPS):
        return ""
    ready = (metric_ready and plan_ready) if ontology_enabled else schema_ready
    if ready and index == before and index < len(plan):
        return "口径和规划已经成功。下一回复只调用 run_sql，执行这一步：" + plan[index]["action"] + "。不要再读本体，不要改计划。"
    return ""


def _hold_queries(plan: list[dict], index: int, evidences: list) -> bool:
    """没有未执行步骤，或已经做满 10 步。这一回复不提供查询。"""
    if not plan or not evidences:
        return False
    return index >= PLAN_STEPS or index >= len(plan)


def _stop_if_idle(stall, index, messages, plan_changed: bool = False) -> bool:
    mark = stall(index, plan_changed)
    if mark == "stop":
        return True
    if mark:
        messages.append({"role": "user", "content": mark})
    return False


def _revise_plan(plan: list[dict], index: int, fresh: list[dict], cap: int) -> tuple[list[dict], str]:
    """用新步骤替换尚未执行的尾巴。可以新增、改写或删除，总数不超过 cap。已完成步骤不重新排队。"""
    done = plan[:index]
    pending = plan[index:]
    if not fresh:
        new_plan = _renumber(done)
        if not new_plan[index:]:
            return new_plan, _tail_done_note(index)
        return new_plan, "计划已修订。尚未执行的步骤：" + _join_actions(new_plan[index:])

    seen = {_norm_action(s["action"]) for s in done}
    novel: list[dict] = []
    for step in fresh:
        key = _norm_action(step["action"])
        if not key or key in seen:
            continue
        seen.add(key)
        novel.append(step)
    pending_keys = [_norm_action(s["action"]) for s in pending]
    novel_keys = [_norm_action(s["action"]) for s in novel]
    if not novel or novel_keys == pending_keys:
        if not pending:
            return plan, _tail_done_note(index)
        if not novel:
            return plan, "这些步骤已经完成，计划不改。尚未执行的步骤：" + _join_actions(pending)
        return plan, "计划没有变化。请继续执行尚未完成的步骤：" + _join_actions(pending)
    room = max(cap - len(done), 0)
    if len(novel) > room:
        if pending:
            return plan, f"计划最多 {cap} 步，不能再新增。尚未执行的步骤仍是：" + _join_actions(pending)
        return plan, f"计划最多 {cap} 步，不能再新增。\n" + _tail_done_note(index)
    new_plan = _renumber(done + novel)
    new_pending = new_plan[len(done):]
    if not new_pending:
        return new_plan, _tail_done_note(len(done))
    return new_plan, "计划已修订。尚未执行的步骤：" + _join_actions(new_pending)


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
        f"【原始问题】{question}\n【分析目标】{goal}\n{_GOAL_SCOPE}\n"
        f"【你的专家方法（Skill）】\n{skill_card}\n"
        + (f"\n{known_block}" if known_block else "")
        + "【任务要求】先给出本视角的取数计划，再按计划调用工具并执行；只基于真实 SQL 结果下结论。"
    )
    await emit({"type": "agent_prompt", "skill_id": skill_id, "prompt": prompt_text})
    opening = "\n".join([
        f"原始问题：{question}",
        f"分析目标：{goal}",
        _GOAL_SCOPE,
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
    idle = 0
    progress_at = 0

    def stall(at: int, extended: bool = False) -> str | None:
        nonlocal idle, progress_at
        # 新的成功查询才算进展。只改尚未执行的步骤不清零。
        # 计划做完后又追加了步骤，下一回复可以查，这一轮不算空转。
        # 计划做完后既不追加也不写结论，计入空转。
        if at > progress_at:
            progress_at = at
            idle = 0
            return None
        if extended:
            idle = 0
            return None
        idle += 1
        if idle >= SQL_IDLE_STOP:
            return "stop"
        if idle != SQL_IDLE_NUDGE or at >= len(plan) or at >= PLAN_STEPS:
            return None
        action = plan[at]["action"]
        ready = (metric_ready and plan_ready) if ontology_enabled else schema_ready
        if ready:
            return "口径和规划已经有了。下一回复只调用 run_sql，执行这一步：" + action + "。不要再读本体。"
        return "下一回复先取得这一步的口径和规划，拿到之后立刻写 SQL。不要重复阅读已经返回过的内容。"

    def stalled():
        return SubResult(
            skill_id=skill_id, skill_name=skill_name, conclusion=_STALLED,
            evidences=evidences, status="truncated",
        ), usage

    for step in range(1, MAX_STEPS + 1):
        await emit({
            "type": "agent_progress", "skill_id": skill_id,
            "skill_name": skill_name, "step": step,
        })
        offered = None if phase == "plan" or _hold_queries(plan, index, evidences) else worker_tools(ontology_enabled)
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
            if len(parsed) > PLAN_STEPS:
                messages.append({
                    "role": "user",
                    "content": (
                        f"计划最多 {PLAN_STEPS} 步，这一份有 {len(parsed)} 步，整份不收下。"
                        f"请收成这道题真正要回答的取数，控制在 {PLAN_SOFT} 步以内，再提交一次 ```plan。"
                    ),
                })
                continue
            plan = parsed
            await emit({"type": "worker_plan", "skill_id": skill_id, "thought": "", "steps": plan})
            phase = "exec"
            index = 0
            messages.append({
                "role": "user",
                "content": "计划已记录。现在用函数执行。一次回复最多 4 个函数。写 SQL 前先调用函数取得这一步需要的口径和规划。口径和规划已经成功之后，下一回复就调用 run_sql，不要再重复阅读本体。一次查询对应一步计划，不要在同一次回复里多带查询。不要在正文里写工具参数。",
            })
            continue

        has_plan = "```plan" in content.lower()
        tools = calls_to_pending(tool_calls)
        taken = _ready_conclusion(content, evidences, plan, index)
        if taken:
            conclusion = taken
            break
        locked = bool(evidences) and index >= PLAN_STEPS
        held = _hold_queries(plan, index, evidences)
        if locked:
            if tools:
                _append_tool_results(messages, [
                    (str(tool.get("call_id") or ""), "未执行：" + _LOCK) for tool in tools
                ], "")
            elif has_plan:
                messages.append({"role": "user", "content": _LOCK})
            else:
                messages.append({"role": "user", "content": _LOCK + "正文里的查询或工具调用不是结论。"})
            if _stop_if_idle(stall, index, messages):
                return stalled()
            continue
        if held and has_plan:
            tools = []
        elif held and tools:
            _append_tool_results(messages, [
                (str(tool.get("call_id") or ""), _OVERFLOW) for tool in tools
            ], "")
            if _stop_if_idle(stall, index, messages):
                return stalled()
            continue
        before = index
        if tools and has_plan:
            _metric, plan_ready, schema_ready, formula, index, pieces = await _run_tools(
                tools, skill_id, emit, cursor, goal, plan, index, evidences,
                ontology_enabled, metric_ready, plan_ready, schema_ready, formula,
            )
            metric_ready = _metric
            follow = (
                _exec_note(plan, index, before, ontology_enabled, metric_ready, plan_ready, schema_ready, evidences)
                or "同一次回复里既有工具又有新计划。这一轮只处理工具，计划不改。请在看见结果后再单独修订计划。"
            )
            _append_tool_results(messages, pieces, _batch_note(plan, index, before, evidences, follow))
            if _stop_if_idle(stall, index, messages):
                return stalled()
            continue
        if tools:
            metric_ready, plan_ready, schema_ready, formula, index, pieces = await _run_tools(
                tools, skill_id, emit, cursor, goal, plan, index, evidences,
                ontology_enabled, metric_ready, plan_ready, schema_ready, formula,
            )
            follow = _exec_note(
                plan, index, before, ontology_enabled, metric_ready, plan_ready, schema_ready, evidences,
            )
            _append_tool_results(messages, pieces, _batch_note(plan, index, before, evidences, follow))
            if _stop_if_idle(stall, index, messages):
                return stalled()
            continue
        if has_plan:
            fresh = _plan_steps(content)
            previous = plan
            if fresh is None:
                messages.append({"role": "user", "content": "没有解析到新的 ```plan。未执行步骤保持不变。"})
                extended = False
            else:
                plan, note = _revise_plan(plan, index, fresh, PLAN_STEPS)
                await emit({"type": "worker_plan", "skill_id": skill_id, "thought": "", "steps": plan})
                messages.append({"role": "user", "content": note})
                extended = index >= len(previous) and index < len(plan)
            if _stop_if_idle(stall, index, messages, extended):
                return stalled()
            continue
        ready = (metric_ready and plan_ready) if ontology_enabled else schema_ready
        action = plan[index]["action"] if index < len(plan) else ""
        if evidences and index >= len(plan):
            nudge = _tail_done_note(index)
        elif ready and action:
            nudge = "口径和规划已经成功。下一回复只调用 run_sql，执行这一步：" + action + "。不要再读本体。"
        else:
            nudge = (
                "这一步需要调用函数，或在看见结果后单独用 ```plan 新增、改写或删除未执行步骤，或在已有 SQL 结果后输出 ```conclusion。"
                f"一次最多 4 个函数。计划控制在 {PLAN_SOFT} 步以内，最多 {PLAN_STEPS} 步。一次查询对应一步计划。不要在正文里写工具参数。"
            )
        messages.append({"role": "user", "content": nudge})
        if _stop_if_idle(stall, index, messages):
            return stalled()

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
            if index >= len(plan) or index >= PLAN_STEPS:
                body = "未执行：" + _LOCK if index >= PLAN_STEPS else _OVERFLOW
                pieces.append((call_id, body))
                await emit({
                    "type": "tool_call", "skill_id": skill_id, "tool": "run_sql",
                    "title": "未执行", "detail": "没有未执行的计划步骤", "body": label(tool),
                })
                continue
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
