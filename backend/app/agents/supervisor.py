"""兼容入口：主编排已改为 LangGraph（见 orchestrator）。"""
from .executors import run_query_point
from .merge import merge_results
from .orchestrator import orchestrate

__all__ = ["orchestrate", "run_query_point", "merge_results"]
