"""run_sql 工具：DuckDB 只读执行，预检 + 快照。"""
from __future__ import annotations
import re
from decimal import Decimal
from datetime import date, datetime
from typing import Any

from ..db import new_cursor, ALIAS_DOC

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH|TRUNCATE|GRANT|REVOKE|PRAGMA|SET|USE)\b",
    re.IGNORECASE,
)


def _jsonable(value: Any) -> Any:
    """DuckDB 非 JSON 原生类型清洗：Decimal→float，date/datetime→ISO 字符串。"""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def run_sql(sql: str, cursor=None) -> dict[str, Any]:
    """只读执行单条 SELECT。返回全部行，行数上限由回传处统一裁。"""
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return {"error": "空 SQL", "columns": [], "rows": [], "row_count": 0, "sql": sql}

    # 多语句拦截
    if ";" in sql:
        return {"error": "拒绝多语句 SQL", "columns": [], "rows": [], "row_count": 0, "sql": sql}
    # 写操作/非 SELECT 拦截
    head = re.sub(r"^\s*with\s+[\s\S]+?\)\s*", "", sql, flags=re.IGNORECASE)
    if _FORBIDDEN.search(sql):
        return {"error": "检测到写操作/禁止关键字，仅允许 SELECT", "columns": [], "rows": [], "row_count": 0, "sql": sql}
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        return {"error": "仅允许 SELECT 查询", "columns": [], "rows": [], "row_count": 0, "sql": sql}

    cur = cursor or new_cursor()
    try:
        res = cur.execute(sql)
        cols = [d[0] for d in res.description]
        rows = [{c: _jsonable(v) for c, v in zip(cols, r)} for r in res.fetchall()]
        return {"columns": cols, "rows": rows, "row_count": len(rows), "sql": sql, "error": None}
    except Exception as e:  # 语法/join 错误回喂给 LLM 修正，不降级
        return {"error": f"{type(e).__name__}: {e}", "columns": [], "rows": [], "row_count": 0, "sql": sql}


def alias_help() -> str:
    return "可用表别名：" + "；".join(f"{k}={v}" for k, v in ALIAS_DOC.items())
