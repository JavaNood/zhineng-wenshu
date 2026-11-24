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
    assert len(QUERY_SYS) == 580 and QUERY_SYS.endswith("{data_map}")
    assert len(WORKER_SYS) == 634 and WORKER_SYS.endswith("{data_map}")
    assert len(MERGE_SYS) == 232 and MERGE_SYS.startswith("你是结果汇聚者")
    assert len(PLANNER_SYS) == 960 and PLANNER_SYS.startswith("你是智能问数系统的规划器")
    assert MAIN_SYS.startswith("你是智能问数的主 Agent")


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
    assert {"main", "ontology", "skills", "sql", "report", "nudge", "worker",
            "collect_workers", "finish"} <= names
    rnames = set(react_graph.REACT.get_graph().nodes)
    assert {"agent", "sql", "reject", "ask", "truncate"} <= rnames


def _tool_block(preface: str, payload: dict) -> str:
    return preface + "\n```tool\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


class _Fake:
    def __init__(self):
        self.worker_users: list[str] = []
        self.main_users: list[str] = []
        self.calls = 0

    async def __call__(self, messages, model=None, on_reasoning=None, on_content=None):
        self.calls += 1
        sys_msg = messages[0]["content"] if messages else ""
        last = messages[-1]["content"] if messages else ""
        usage = {"prompt": 2, "completion": 1, "total": 3}
        if on_reasoning and sys_msg.startswith("你是智能问数的主 Agent"):
            await on_reasoning("本步思考。")
        if sys_msg.startswith("你是智能问数的主 Agent"):
            question = messages[1]["content"]
            self.main_users.append(last)
            return {"content": self._main(question, last), "reasoning": "", "usage": usage}
        if sys_msg.startswith("你是一位被派遣的分析专家"):
            self.worker_users.append(last)
            return {"content": self._worker(last), "reasoning": "", "usage": usage}
        return {"content": "", "reasoning": "", "usage": usage}

    def _main(self, question: str, last: str) -> str:
        why = any(w in question for w in ("为什么", "原因", "下降", "组合"))
        if "你好" in question and not last.startswith("工具"):
            return _tool_block("先直接回答。", {"tool": "finish", "summary": "你好，需要问数直接说。"})
        if not last.startswith("工具"):
            return _tool_block("先看销售额口径。", {"tool": "read_ontology", "metric": "销售额"})
        if last.startswith("工具 read_ontology"):
            return _tool_block(
                "按本体口径查销售额。",
                {"tool": "run_sql", "sql": "SELECT SUM(实付金额)-SUM(退款金额) AS sales FROM orders"},
            )
        if last.startswith("工具 run_sql"):
            return _tool_block("把查到的数值告诉用户。", {"tool": "report", "text": "销售额已取到。"})
        if last.startswith("工具 report"):
            if why:
                return _tool_block("数值已经有了，接下来看有哪些专家。", {"tool": "list_skills"})
            return _tool_block("数值已经回答完。", {"tool": "finish", "summary": "销售额已取到。"})
        if last.startswith("工具 list_skills"):
            skills = (
                ["value-price-breakdown", "seasonality-diagnosis"]
                if "组合" in question
                else ["value-price-breakdown", "customer-refund"]
            )
            return _tool_block(
                "选两个互补的专家来解释原因。",
                {"tool": "dispatch", "skills": skills, "goal": "解释销售额下降"},
            )
        if last.startswith("工具 dispatch"):
            return _tool_block("专家结论已经齐了，开始汇总。", {
                "tool": "finish",
                "summary": "两位专家都指向结构变化。",
                "consensus": [{"point": "结构变化", "support": ["value-price-breakdown", "customer-refund"]}],
                "divergences": [],
            })
        return _tool_block("先看销售额口径。", {"tool": "read_ontology", "metric": "销售额"})

    def _worker(self, last: str) -> str:
        if "先制定计划" in last:
            return (
                "```plan\n"
                '{"steps":[{"n":1,"action":"按渠道对比两期销售额并计算差额"},'
                '{"n":2,"action":"按门店列出销售额下降最多的前十家"}]}\n```'
            )
        if "计划中的取数都已完成" in last:
            return "```conclusion\n销售额见上方查询。\n## 该专家的主要结论\n销售额已取到。\n```"
        return "```sql\nSELECT SUM(实付金额)-SUM(退款金额) AS sales FROM orders\n```"


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
    assert types.count("chat_answer") == 1
    assert types.count("tokens") == 1
    assert "query_answer" not in types
    assert "final" not in types
    assert "agent_start" not in types
    assert events[types.index("chat_answer")]["text"] == "你好，需要问数直接说。"
    assert session.tokens["total"] > 0
    assert session.sub_results == []


async def _merged_query(fake: _Fake) -> None:
    events, _session = await _run("请看两年销售额", fake)
    assert sum(1 for e in events if e["type"] == "query_answer") == 1
    onto_at = next(i for i, e in enumerate(events)
                   if e["type"] == "tool_call" and e.get("tool") == "read_ontology")
    sql_at = next(i for i, e in enumerate(events)
                  if e["type"] == "tool_call" and e.get("tool") == "run_sql" and e.get("skill_id") == "main")
    assert events[onto_at]["skill_id"] == "main"
    assert "销售额" in (events[onto_at].get("body") or "")
    assert onto_at < sql_at
    assert any(e["type"] == "reasoning" and i < onto_at for i, e in enumerate(events))
    assert any(e["type"] == "act" and i < onto_at for i, e in enumerate(events))
    answer = next(e for e in events if e["type"] == "query_answer")
    assert "销售额" in answer["text"]
    assert answer["evidences"]
    assert not any(e["type"] == "agent_start" for e in events)


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
    onto_at = next(i for i, e in enumerate(events) if e["type"] == "tool_call" and e.get("tool") == "read_ontology")
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

    async def think(messages, model=None, on_reasoning=None, on_content=None):
        calls["n"] += 1
        return {"content": "仍在思考下一步。", "reasoning": "",
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
    assert calls["n"] == MAX_STEPS == 60
    assert sub.status == "truncated"
    assert sub.conclusion == "达到 step 上限，结论不完整（已中止，不伪造完成）。"
    assert sub.evidences == []
    assert sum(1 for e in events if e["type"] == "step_log") == 0
    assert usage["total"] == 120


async def main() -> None:
    _prompts_unchanged()
    _groups()
    _graph_shape()
    init_db()
    original = llm.chat_stream
    try:
        await _chat(_Fake())
        await _merged_query(_Fake())
        await _analysis(_Fake())
        await _query_then_analysis(_Fake())
        await _isolated_failure(_Fake())
        await _step_cap()
    finally:
        llm.chat_stream = original
    print("GRAPH SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())
