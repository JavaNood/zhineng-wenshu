"""LangGraph 编排结构测试：不调真实 LLM。

校验主图节点、问数点合并、专家并行汇聚、单专家异常隔离、step 上限，以及提示词正文长度未变。
"""
from __future__ import annotations
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app import llm
from app.agents import executors, orchestrator, react_graph
from app.agents.main_react import MAIN_SYS
from app.agents.merge import MERGE_SYS
from app.agents.orchestrator import build_groups, orchestrate
from app.agents.planner import PLANNER_SYS
from app.agents.react_graph import QUERY_SYS, WORKER_SYS
from app.agents.worker import run_skill_worker
from app.config import MAX_PARALLEL, MAX_STEPS
from app.db import init_db
from app.schemas import IntentPoint, Plan
from app.store.session import create_session
from app.tools import sql_tool


def _prompts_unchanged() -> None:
    assert "data_map" not in QUERY_SYS and "口径以工具返回为准" in QUERY_SYS
    assert WORKER_SYS.startswith("你是一位被派遣的分析专家") and "data_map" not in WORKER_SYS
    assert "resolve_certified_metric" in WORKER_SYS and "一次回复最多调用 4 个函数" in WORKER_SYS
    assert "```tools" not in WORKER_SYS and "```tools" not in MAIN_SYS
    assert "可以新增、改写或删除" in WORKER_SYS
    assert "控制在 6 步以内，最多 10 步" in WORKER_SYS
    assert "不是你的任务清单" in WORKER_SYS
    assert "不要把某一位专家的拆法写进 goal" in MAIN_SYS
    assert "不能让总步数超过第一次收下的计划" not in WORKER_SYS
    assert "run_sql" in QUERY_SYS and "```sql" not in QUERY_SYS
    assert len(MERGE_SYS) == 232 and MERGE_SYS.startswith("你是结果汇聚者")
    assert len(PLANNER_SYS) == 960 and PLANNER_SYS.startswith("你是智能问数系统的规划器")
    assert MAIN_SYS.startswith("你是智能问数的主 Agent")
    assert "一次回复最多调用 4 个函数" in MAIN_SYS


def _groups() -> None:
    points = [
        IntentPoint(0, "a", "query", Plan("query", "2025年销售额")),
        IntentPoint(1, "b", "query", Plan("query", "2024年销售额")),
        IntentPoint(2, "c", "chat", Plan("chat", "寒暄")),
        IntentPoint(3, "d", "query", Plan("query", "订单量")),
        IntentPoint(4, "e", "analysis", Plan(
            "analysis", "解释下降", skill_ids=["value-price-breakdown", "customer-refund"])),
    ]
    groups = build_groups(points, "原问题")
    assert [g["kind"] for g in groups] == ["query", "chat", "query", "analysis"]
    assert groups[0]["goal"] == "2025年销售额；2024年销售额"
    assert groups[3]["skill_ids"] == ["value-price-breakdown", "customer-refund"]


def _graph_shape() -> None:
    names = set(orchestrator.GRAPH.get_graph().nodes)
    assert {"main", "tools", "nudge", "worker", "collect_workers", "finish"} <= names
    assert "ontology" not in names
    rnames = set(react_graph.REACT.get_graph().nodes)
    assert {"agent", "sql", "reject", "ask", "truncate"} <= rnames


def _call(fake_seq: list[int], name: str, arguments: dict) -> dict:
    fake_seq[0] += 1
    return {
        "id": f"call_{fake_seq[0]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _names(messages: list[dict]) -> list[str]:
    names = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            names.append((call.get("function") or {}).get("name") or "")
    return names


def _last_user(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


class _Fake:
    def __init__(self):
        self.worker_users: list[str] = []
        self.main_users: list[str] = []
        self.calls = 0
        self._seq = [0]

    async def __call__(self, messages, model=None, on_reasoning=None, on_content=None, tools=None):
        self.calls += 1
        sys_msg = messages[0]["content"] if messages else ""
        last = messages[-1].get("content") or "" if messages else ""
        usage = {"prompt": 2, "completion": 1, "total": 3}
        if on_reasoning and sys_msg.startswith("你是智能问数的主 Agent"):
            await on_reasoning("本步思考。")
        if sys_msg.startswith("你是智能问数的主 Agent"):
            assert tools and any(item["function"]["name"] == "run_sql" for item in tools)
            question = messages[1]["content"]
            self.main_users.append(last)
            content, calls = self._main(question, messages)
            return {"content": content, "reasoning": "本步思考。", "tool_calls": calls, "usage": usage}
        if sys_msg.startswith("你是一位被派遣的分析专家"):
            user_blob = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "user")
            blob = "\n".join(
                m.get("content") or "" for m in messages if m.get("role") != "system"
            )
            if "取数都已完成" in blob:
                assert not tools
            elif "计划已记录" in user_blob or "计划已修订" in user_blob:
                assert tools, "执行阶段必须带上 tools"
            else:
                assert not tools
            self.worker_users.append(last)
            content, calls = self._worker(messages)
            return {"content": content, "reasoning": "", "tool_calls": calls, "usage": usage}
        return {"content": "", "reasoning": "", "tool_calls": [], "usage": usage}

    def _main(self, question: str, messages: list[dict]) -> tuple[str, list[dict]]:
        why = any(w in question for w in ("为什么", "原因", "下降", "组合"))
        names = _names(messages)
        last_user = _last_user(messages)
        if last_user.startswith("工具 dispatch"):
            return "专家结论已经齐了，开始汇总。", [_call(self._seq, "finish", {
                "summary": "两位专家都指向结构变化。",
                "consensus": [{"point": "结构变化", "support": ["value-price-breakdown", "customer-refund"]}],
                "divergences": [],
            })]
        if "你好" in question and not names:
            return "你好，需要问数直接说。", [_call(self._seq, "finish", {"summary": "你好，需要问数直接说。"})]
        if "resolve_certified_metric" not in names:
            return "先对齐认证指标和 SQL 规划。", [
                _call(self._seq, "resolve_certified_metric", {"metric": "销售额"}),
                _call(self._seq, "read_sql_plan", {"metric": "销售额"}),
            ]
        if "run_sql" not in names:
            return "按认证口径查销售额。", [_call(self._seq, "run_sql", {
                "sql": "SELECT SUM(实付金额)-SUM(退款金额) AS sales FROM orders",
            })]
        if "report" not in names:
            return "把查到的数值告诉用户。", [_call(self._seq, "report", {"text": "销售额已取到。"})]
        if why and "list_skills" not in names:
            return "数值已经有了，接下来看有哪些专家。", [_call(self._seq, "list_skills", {})]
        if why and "dispatch" not in names:
            skills = (
                ["value-price-breakdown", "seasonality-diagnosis"]
                if "组合" in question
                else ["value-price-breakdown", "customer-refund"]
            )
            return "选两个互补的专家来解释原因。", [_call(self._seq, "dispatch", {
                "skills": skills, "goal": "解释销售额下降",
            })]
        return "数值已经回答完。", [_call(self._seq, "finish", {"summary": "销售额已取到。"})]

    def _worker(self, messages: list[dict]) -> tuple[str, list[dict]]:
        last = messages[-1].get("content") or ""
        if messages[-1].get("role") == "user" and "先制定计划" in last:
            return (
                "```plan\n"
                '{"steps":[{"n":1,"action":"按渠道对比两期销售额并计算差额"},'
                '{"n":2,"action":"按门店列出销售额下降最多的前十家"}]}\n```'
            ), []
        if any("计划中的取数都已完成" in (m.get("content") or "") for m in messages):
            return "```conclusion\n销售额见上方查询。\n## 该专家的主要结论\n销售额已取到。\n```", []
        names = _names(messages)
        if "resolve_certified_metric" not in names:
            return "先取口径和规划。", [
                _call(self._seq, "resolve_certified_metric", {"metric": "销售额"}),
                _call(self._seq, "read_sql_plan", {"metric": "销售额"}),
            ]
        return "按计划查这一步。", [_call(self._seq, "run_sql", {
            "sql": "SELECT SUM(实付金额)-SUM(退款金额) AS sales FROM orders",
        })]


async def _run(question: str, fake: _Fake) -> tuple[list[dict], object]:
    llm.chat_stream = fake
    seen = {}
    original = orchestrator.GRAPH.ainvoke

    async def wrapped(state, config=None, **kwargs):
        seen["config"] = config
        seen["context"] = kwargs.get("context")
        return await original(state, config=config, **kwargs)

    orchestrator.GRAPH.ainvoke = wrapped
    events: list[dict] = []

    async def emit(ev: dict):
        events.append(ev)

    try:
        session = create_session(question, "deepseek-chat")
        await orchestrate(question, "deepseek-chat", emit, session)
    finally:
        orchestrator.GRAPH.ainvoke = original
    assert seen["config"]["max_concurrency"] == MAX_PARALLEL == 3
    assert seen["config"]["recursion_limit"] == orchestrator.OUTER_RECURSION_LIMIT
    assert set(seen["context"]) == {"emit", "session"}
    assert not any(e["type"] == "error" for e in events)
    blob = json.dumps(events, ensure_ascii=False)
    for banned in (
        "数值部分我已经直接查到",
        "不固定派齐",
        "位专家已完成",
        "形成查询结论",
        "形成分析结论",
        "通读可用分析本领",
        "执行查询 · run_sql",
        "查阅本体 · 对齐指标口径",
    ):
        assert banned not in blob, banned
    return events, session


async def _chat(fake: _Fake) -> None:
    events, session = await _run("你好", fake)
    types = [e["type"] for e in events]
    assert "chat_answer" not in types
    assert types.count("tokens") == 1
    assert "query_answer" not in types
    assert "final" not in types
    assert "agent_start" not in types
    assert any(e["type"] == "act" and e.get("text") == "你好，需要问数直接说。" for e in events)
    assert not any(e["type"] == "tool_call" and e.get("tool") == "finish" for e in events)
    assert session.tokens["total"] > 0
    assert session.sub_results == []


async def _merged_query(fake: _Fake) -> None:
    events, _session = await _run("请看两年销售额", fake)
    assert sum(1 for e in events if e["type"] == "query_answer") == 1
    onto_at = next(i for i, e in enumerate(events)
                   if e["type"] == "tool_call" and e.get("tool") == "resolve_certified_metric"
                   and e.get("skill_id") == "main")
    sql_at = next(i for i, e in enumerate(events)
                  if e["type"] == "tool_call" and e.get("tool") == "run_sql" and e.get("skill_id") == "main")
    assert events[onto_at]["skill_id"] == "main"
    assert "销售额" in (events[onto_at].get("body") or "")
    assert onto_at < sql_at
    assert any(e["type"] == "reasoning" and i < onto_at for i, e in enumerate(events))
    assert any(e["type"] == "act" and i < onto_at for i, e in enumerate(events))
    answer_at = next(i for i, e in enumerate(events) if e["type"] == "query_answer")
    answer = events[answer_at]
    assert "销售额" in answer["text"]
    assert answer["evidences"]
    assert not any(e["type"] == "agent_start" for e in events)
    assert not any(e["type"] == "reasoning" and i > answer_at for i, e in enumerate(events))
    assert events[answer_at + 1]["type"] == "tokens"


async def _analysis(fake: _Fake) -> None:
    events, session = await _run("销售额为什么下降", fake)
    starts = [e["skill_id"] for e in events if e["type"] == "agent_start"]
    assert starts == ["value-price-breakdown", "customer-refund"] or set(starts) == {
        "value-price-breakdown", "customer-refund"}
    assert len(starts) == 2
    assert [r["skill_id"] for r in session.sub_results] == [
        "value-price-breakdown", "customer-refund"]
    assert all(r["status"] == "done" for r in session.sub_results)
    texts = "".join(e.get("text") or "" for e in events if e["type"] == "reasoning")
    assert "本步思考。" in texts
    skills_at = next(i for i, e in enumerate(events) if e["type"] == "tool_call" and e.get("tool") == "list_skills")
    dispatch_at = next(i for i, e in enumerate(events) if e["type"] == "tool_call" and e.get("tool") == "dispatch")
    assert skills_at < dispatch_at
    done_at = max(i for i, e in enumerate(events) if e["type"] == "sub_done")
    final_at = next(i for i, e in enumerate(events) if e["type"] == "final")
    assert any(e["type"] == "act" and "汇总" in (e.get("text") or "") and done_at < i < final_at
               for i, e in enumerate(events))
    assert any(e["type"] == "reasoning" and done_at < i < final_at for i, e in enumerate(events))
    for sid in ("value-price-breakdown", "customer-refund"):
        plan = next(e for e in events if e["type"] == "worker_plan" and e["skill_id"] == sid)
        sqls = [e for e in events if e["type"] == "step_log" and e["skill_id"] == sid and e["kind"] == "sql"]
        assert [s["action"] for s in plan["steps"]] == [e["action"] for e in sqls]
        assert plan["steps"][0]["action"].startswith("按渠道对比")
    assert not any(e["type"] == "step_log" and e.get("kind") == "think" for e in events)
    obs = next(u for u in reversed(fake.main_users) if u.startswith("工具 dispatch"))
    assert obs.find("量价") < obs.find("客群")
    assert any(e["type"] == "consensus" for e in events)
    final = next(e for e in events if e["type"] == "final")
    assert final["summary"] == "两位专家都指向结构变化。"
    assert [e["type"] for e in events].count("tokens") == 1
    assert session.final and session.final["summary"] == final["summary"]


async def _query_then_analysis(fake: _Fake) -> None:
    events, session = await _run("组合问题：销售额以及为什么下降", fake)
    initials = [u for u in fake.worker_users if u.startswith("原始问题")]
    assert len(initials) == 2
    assert all("销售额已取到" in u and "禁止重复聚合同一总量" in u for u in initials)
    assert [r["skill_id"] for r in session.sub_results] == [
        "value-price-breakdown", "seasonality-diagnosis"]
    q_at = next(i for i, e in enumerate(events) if e["type"] == "query_answer")
    a_at = next(i for i, e in enumerate(events) if e["type"] == "agent_start")
    onto_at = next(i for i, e in enumerate(events)
                   if e["type"] == "tool_call" and e.get("tool") == "resolve_certified_metric"
                   and e.get("skill_id") == "main")
    assert onto_at < q_at < a_at


async def _isolated_failure(fake: _Fake) -> None:
    real = executors.run_skill_worker

    async def wrapped(*args, **kwargs):
        if args and args[0] == "customer-refund":
            raise RuntimeError("boom")
        return await real(*args, **kwargs)

    executors.run_skill_worker = wrapped
    try:
        events, session = await _run("销售额为什么下降需要诊断", fake)
    finally:
        executors.run_skill_worker = real
    assert [r["skill_id"] for r in session.sub_results] == [
        "value-price-breakdown", "customer-refund"]
    assert session.sub_results[0]["status"] == "done"
    assert session.sub_results[1]["status"] == "error"
    assert "boom" in session.sub_results[1]["conclusion"]
    assert any(e["type"] == "final" for e in events)


async def _step_cap() -> None:
    calls = {"n": 0}

    async def think(messages, model=None, on_reasoning=None, on_content=None, tools=None):
        calls["n"] += 1
        return {"content": "仍在思考下一步。", "reasoning": "", "tool_calls": [],
                "usage": {"prompt": 1, "completion": 1, "total": 2}}

    llm.chat_stream = think
    events: list[dict] = []

    async def emit(ev: dict):
        events.append(ev)

    init_db()
    sub, usage = await run_skill_worker(
        "value-price-breakdown", "解释下降", "为什么下降", "deepseek-chat",
        emit, sql_tool.new_cursor(),
    )
    assert calls["n"] == MAX_STEPS == 30
    assert sub.status == "truncated"
    assert sub.conclusion == "达到 step 上限，结论不完整（已中止，不伪造完成）。"
    assert sub.evidences == []
    assert sum(1 for e in events if e["type"] == "step_log") == 0
    assert usage["total"] == 60


async def _query_react() -> None:
    async def emit(_ev):
        return None

    async def fake(messages, model=None, on_reasoning=None, on_content=None, tools=None):
        assert tools and [item["function"]["name"] for item in tools] == ["run_sql"]
        names = _names(messages)
        if "run_sql" not in names:
            return {
                "content": "先查一行。",
                "reasoning": "想一下。",
                "tool_calls": [_call([0], "run_sql", {"sql": "SELECT 1 AS n"})],
                "usage": {"prompt": 1, "completion": 1, "total": 2},
            }
        assert any(m.get("role") == "tool" for m in messages)
        assistant = next(m for m in messages if m.get("role") == "assistant")
        assert "reasoning_content" in assistant
        assert assistant.get("tool_calls")
        return {
            "content": "```conclusion\n结果是 1。\n```",
            "reasoning": "可以收口。",
            "tool_calls": [],
            "usage": {"prompt": 1, "completion": 1, "total": 2},
        }

    llm.chat_stream = fake
    sub, _usage = await executors.run_query_point(
        "一行", "查一行", "deepseek-chat", emit, sql_tool.new_cursor(),
    )
    assert sub.status == "done"
    assert "结果是 1" in sub.conclusion
    assert sub.evidences and sub.evidences[0].row_count == 1


def _plan_revision() -> None:
    from app.agents.plan_execute import _revise_plan

    plan = [
        {"n": 1, "action": "按客户等级对比两期销售额"},
        {"n": 2, "action": "按是否新客首单对比两期销售额"},
        {"n": 3, "action": "按是否企业客户对比两期销售额"},
        {"n": 4, "action": "按订单状态对比两期退款金额"},
    ]
    cap = 20
    grown = plan + [{"n": 5, "action": "按促销标记交叉客户等级"}]
    extended, note = _revise_plan(plan, 4, grown, cap)
    assert [s["action"] for s in extended] == [s["action"] for s in plan] + ["按促销标记交叉客户等级"]
    assert note.startswith("计划已修订")
    assert "按促销标记交叉客户等级" in note

    same, note = _revise_plan(plan, 2, plan, cap)
    assert [s["action"] for s in same] == [s["action"] for s in plan]
    assert "没有变化" in note

    rewritten = [
        {"n": 1, "action": "按客户等级对比两期销售额"},
        {"n": 2, "action": "按是否新客首单对比两期销售额"},
        {"n": 3, "action": "改写后的企业客户对照"},
        {"n": 4, "action": "改写后的退款对照"},
    ]
    revised, note = _revise_plan(plan, 2, rewritten, cap)
    assert [s["action"] for s in revised] == [
        "按客户等级对比两期销售额",
        "按是否新客首单对比两期销售额",
        "改写后的企业客户对照",
        "改写后的退款对照",
    ]
    assert note.startswith("计划已修订")

    longer = rewritten + [{"n": 5, "action": "再加一步促销交叉"}]
    added, note = _revise_plan(plan, 2, longer, cap)
    assert [s["action"] for s in added] == [
        "按客户等级对比两期销售额",
        "按是否新客首单对比两期销售额",
        "改写后的企业客户对照",
        "改写后的退款对照",
        "再加一步促销交叉",
    ]
    assert note.startswith("计划已修订")

    held, note = _revise_plan(plan, 2, longer, 4)
    assert held == plan
    assert "不能再新增" in note
    assert "按是否企业客户对比两期销售额" in note

    dropped, note = _revise_plan(plan, 2, [], cap)
    assert [s["action"] for s in dropped] == [s["action"] for s in plan[:2]]
    assert "取数都已完成" in note

    from app.agents.plan_execute import PLAN_STEPS, _plan_steps, _take_conclusion
    assert PLAN_STEPS == 10
    too_long = "```plan\n{\"steps\":[" + ",".join(
        '{"n":%d,"action":"按维度%d对比两期销售额"}' % (i, i) for i in range(1, 12)
    ) + "]}\n```"
    parsed = _plan_steps(too_long)
    assert parsed is not None and len(parsed) == 11
    held_all, note = _revise_plan(plan, 0, parsed, PLAN_STEPS)
    assert held_all == plan
    assert "不能再新增" in note
    dsml = '<｜｜DSML｜｜ invoke name="run_sql"> SELECT 1'
    assert _take_conclusion(dsml, True) == ""
    assert _take_conclusion("SELECT 1", True) == ""
    assert "团购" in _take_conclusion(
        "```conclusion\n增长来自企业团购。\n## 该专家的主要结论\n增长来自团购。\n```",
        False,
    )
    mixed = (
        "量价几乎各占一半，订单量与客单价同向小幅上升，没有单一因素主导。"
        "\n## 该专家的主要结论\n增长来自企业团购。\n"
        "```plan\n{\"steps\":[{\"n\":6,\"action\":\"再按渠道拆开两期销售额\"}]}\n```"
    )
    assert "企业团购" in _take_conclusion(mixed, True)
    assert _take_conclusion(mixed, False) == ""


async def main() -> None:
    _prompts_unchanged()
    _groups()
    _graph_shape()
    _plan_revision()
    init_db()
    original = llm.chat_stream
    try:
        await _chat(_Fake())
        await _merged_query(_Fake())
        await _analysis(_Fake())
        await _query_then_analysis(_Fake())
        await _isolated_failure(_Fake())
        await _step_cap()
        await _query_react()
    finally:
        llm.chat_stream = original
    print("GRAPH SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())
