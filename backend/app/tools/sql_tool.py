"""run_sql 工具：DuckDB 只读执行，预检 + 快照。"""
from __future__ import annotations
import re
from decimal import Decimal
from datetime import date, datetime
from typing import Any

from ..db import new_cursor, ALIAS_DOC, ALIASES

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


def run_sql(sql: str, cursor=None, limit: int = 500) -> dict[str, Any]:
    """只读执行单条 SELECT。返回 {columns, rows, row_count, sql, error}。"""
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
        # 强制 LIMIT 兜底，防止全表拉出
        wrapped = sql if re.search(r"\blimit\s+\d+", sql, re.IGNORECASE) else f"{sql} LIMIT {limit}"
        res = cur.execute(wrapped)
        cols = [d[0] for d in res.description]
        rows = [{c: _jsonable(v) for c, v in zip(cols, r)} for r in res.fetchall()]
        return {"columns": cols, "rows": rows, "row_count": len(rows), "sql": sql, "error": None}
    except Exception as e:  # 语法/join 错误回喂给 LLM 修正，不降级
        return {"error": f"{type(e).__name__}: {e}", "columns": [], "rows": [], "row_count": 0, "sql": sql}


def alias_help() -> str:
    return "可用表别名：" + "；".join(f"{k}={v}" for k, v in ALIAS_DOC.items())


def data_map_text() -> str:
    """给执行 Agent 的紧凑数据地图：表别名→字段清单 + 允许 join 白名单，
    避免 LLM 盲探 information_schema 或误判表无关联字段。"""
    from ..ontology import loader as onto
    ont = onto.load_ontology()
    # 中文表名 -> 英文别名（db.ALIASES：中文 -> (alias, file)）
    cn2alias = {cn: a for cn, (a, _f) in ALIASES.items()}
    lines = ["【数据表与字段（SQL 用英文别名，列名用中文原名）】"]
    for ds in ont.get("datasets", []):
        cn = ds["source"]
        alias = cn2alias.get(cn, cn)
        fields = "、".join(f["name"] for f in ds.get("fields", []))
        lines.append(f"- {alias}（{cn}）：{fields}")
    rels = ont.get("relationships", [])
    if rels:
        lines.append("【允许的 join 白名单（只可走这些路径）】")
        for r in rels:
            lines.append(f"- {r['from']} = {r['to']}（{r.get('type', '')}）")
    lines.append("【日期过滤捷径】orders.订单日期 是 date 型，可直接 WHERE 提取年份/季度"
                 "（如 strftime('%Y', 订单日期)='2025'）；需要促销标记/标准年月时再 join d_date。")
    return "\n".join(lines)
