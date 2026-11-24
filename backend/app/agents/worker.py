"""worker：子 Agent 执行入口。

取数循环已迁入 LangGraph ReAct 子图（react_graph）。
提示词正文在 react_graph.WORKER_SYS，内容与迁移前一致。
"""
from .executors import run_skill_worker
from .textutil import _action_before_sql, _extract_block, _extract_sql, _rows_preview

__all__ = [
    "run_skill_worker",
    "_extract_sql",
    "_extract_block",
    "_rows_preview",
    "_action_before_sql",
]
