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
    assert "你是友好的助手，直接简短回答寒暄与一般问题，不查数。" in open(
        os.path.join(os.path.dirname(__file__), "..", "backend", "app", "agents", "orchestrator.py"),
        encoding="utf-8",
    ).read()


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
    assert {"plan", "chat", "query", "prepare_analysis", "worker",
            "collect_workers", "merge", "finish"} <= names
    rnames = set(react_graph.REACT.get_graph().nodes)
    assert {"agent", "sql", "reject", "ask", "truncate"} <= rnames


class _Fake:
    def __init__(self):
        self.query_users: list[str] = []
        self.worker_users: list[str] = []
        self.merge_users: list[str] = []
        self.calls = 0

    async def __call__(self, messages, model=None, on_reasoning=None, on_content=None):
        self.calls += 1
        sys_msg = messages[0]["content"] if messages else ""
        last = messages[-1]["content"] if messages else ""
        usage = {"prompt": 2, "completion": 1, "total": 3}
        if sys_msg.startswith("你是智能问数系统的规划器"):
            if "组合" in last:
                content = json.dumps({
                    "plan": "先取销售额，再派专家解释。",
                    "points": [
                        {"index": 0, "text": "销售额", "intent": "query", "goal": "2025年销售额",
                         "skills": [], "steps": []},
                        {"index": 1, "text": "为什么下降", "intent": "analysis",
                         "goal": "解释销售额下降",
                         "skills": ["value-price-breakdown", "seasonality-diagnosis"],
                         "steps": []},
                    ],
                }, ensure_ascii=False)
            elif "你好" in last:
                content = json.dumps({
                    "plan": "先打招呼。",
                    "points": [{"index": 0, "text": "你好", "intent": "chat",
                                "goal": "寒暄", "skills": [], "steps": []}],
                }, ensure_ascii=False)
            elif "两年" in last:
                content = json.dumps({
                    "plan": "一次取出两年销售额。",
                    "points": [
                        {"index": 0, "text": "2025", "intent": "query", "goal": "2025年销售额",
                         "skills": [], "steps": [{"step_id": "s1", "action": "查2025", "query_spec": "2025"}]},
                        {"index": 1, "text": "2024", "intent": "query", "goal": "2024年销售额",
                         "skills": [], "steps": []},
                    ],
                }, ensure_ascii=False)
            else:
                content = json.dumps({
                    "plan": "先派两位专家再汇总。",
                    "points": [{
                        "index": 0, "text": "为什么下降", "intent": "analysis",
                        "goal": "解释销售额下降",
                        "skills": ["value-price-breakdown", "customer-refund"],
                        "steps": [{"step_id": "s1", "action": "并行诊断", "query_spec": ""}],
                    }],
                }, ensure_ascii=False)
            return {"content": content, "reasoning": "", "usage": usage}
        if sys_msg.startswith("你是友好的助手"):
            return {"content": "你好，需要问数直接说。", "reasoning": "", "usage": usage}
        if sys_msg.startswith("你是结果汇聚者"):
            self.merge_users.append(last)
            return {"content": json.dumps({
                "summary": "两位专家都指向结构变化。",
                "consensus": [{"point": "结构变化", "support": ["value-price-breakdown", "customer-refund"]}],
                "divergences": [],
            }, ensure_ascii=False), "reasoning": "", "usage": usage}
        if sys_msg.startswith("你是问数执行 Agent"):
            self.query_users.append(last)
        if sys_msg.startswith("你是一位被派遣的分析专家"):
            self.worker_users.append(last)
        if "SQL 结果" in last or last.startswith("结果（"):
            return {"content": "```conclusion\n销售额见上方查询。\n## 该专家的主要结论\n销售额已取到。\n```",
                    "reasoning": "", "usage": usage}
        return {"content": "步骤一：统计销售额\n```sql\nSELECT SUM(实付金额)-SUM(退款金额) AS sales FROM orders\n```",
                "reasoning": "", "usage": usage}


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
    assert sum(1 for e in events if e["type"] == "tool_call" and e.get("tool") == "read_ontology"
               and e.get("skill_id") == "query") == 1
    plan = next(e for e in events if e["type"] == "plan")
    assert len(plan["plan"]["points"]) == 2
    assert any("2025年销售额；2024年销售额" in u for u in fake.query_users)
    answer = next(e for e in events if e["type"] == "query_answer")
    assert "销售额" in answer["text"]
    assert answer["evidences"]


async def _analysis(fake: _Fake) -> None:
    events, session = await _run("销售额为什么下降", fake)
    starts = [e["skill_id"] for e in events if e["type"] == "agent_start"]
    assert starts == ["value-price-breakdown", "customer-refund"] or set(starts) == {
        "value-price-breakdown", "customer-refund"}
    assert len(starts) == 2
    assert [r["skill_id"] for r in session.sub_results] == [
        "value-price-breakdown", "customer-refund"]
    assert all(r["status"] == "done" for r in session.sub_results)
    assert any(e["type"] == "tool_call" and e.get("tool") == "list_skills" for e in events)
    assert any(e["type"] == "consensus" for e in events)
    final = next(e for e in events if e["type"] == "final")
    assert final["summary"] == "两位专家都指向结构变化。"
    assert fake.merge_users and fake.merge_users[-1].find("量价") < fake.merge_users[-1].find("客群")
    assert [e["type"] for e in events].count("tokens") == 1
    assert session.final and session.final["summary"] == final["summary"]


async def _query_then_analysis(fake: _Fake) -> None:
    events, session = await _run("组合问题：销售额以及为什么下降", fake)
    kinds = [p["intent"] for p in next(e for e in events if e["type"] == "plan")["plan"]["points"]]
    assert kinds == ["query", "analysis"]
    initials = [u for u in fake.worker_users if u.startswith("原始问题")]
    assert len(initials) == 2
    assert all("销售额已取到" in u and "禁止重复聚合同一总量" in u for u in initials)
    assert [r["skill_id"] for r in session.sub_results] == [
        "value-price-breakdown", "seasonality-diagnosis"]
    q_at = next(i for i, e in enumerate(events) if e["type"] == "query_answer")
    a_at = next(i for i, e in enumerate(events) if e["type"] == "agent_start")
    assert q_at < a_at


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
    assert sum(1 for e in events if e["type"] == "step_log") == 60
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
