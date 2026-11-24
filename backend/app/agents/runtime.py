"""单次运行的上下文：LangGraph context 传递 emit/session，cursor 按子任务隔离。"""
from __future__ import annotations
import contextvars
from typing import Any, TypedDict

# 每个子 Agent / 问数循环一张独立 cursor，不能放进共享 context（并行时会互相覆盖）。
_cursor: contextvars.ContextVar[Any] = contextvars.ContextVar("qa_sql_cursor", default=None)


class GraphContext(TypedDict):
    emit: Any
    session: Any


def bind_cursor(cursor: Any):
    return _cursor.set(cursor)


def reset_cursor(token) -> None:
    _cursor.reset(token)


def current_cursor():
    return _cursor.get()
