"""本体工具：read_ontology / resolve_metric。"""
from __future__ import annotations
import json
import re
from ..ontology import loader as onto

# 这两处保留业务名和文件名，其余出现的业务表名按本体 sql 字段呈现给模型。
_KEEP = {"source", "file"}


def _sql_pairs() -> list[tuple[str, str]]:
    pairs = []
    for ds in onto.load_ontology().get("datasets", []):
        src, sql = ds.get("source"), ds.get("sql")
        if src and sql:
            pairs.append((src, sql))
    pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
    return pairs


def _rewrite(text: str, pairs: list[tuple[str, str]]) -> str:
    out = text
    for src, sql in pairs:
        out = re.sub(rf"{re.escape(src)}(?!\.csv)", sql, out)
    return out


def _for_llm(node, pairs: list[tuple[str, str]], key: str | None = None):
    if isinstance(node, dict):
        return {k: node[k] if k in _KEEP else _for_llm(node[k], pairs, k) for k in node}
    if isinstance(node, list):
        return [_for_llm(item, pairs, key) for item in node]
    if isinstance(node, str) and key not in _KEEP:
        return _rewrite(node, pairs)
    return node


def read_ontology(topic: str = "") -> str:
    data = _for_llm(onto.read_ontology(topic), _sql_pairs())
    return json.dumps(data, ensure_ascii=False, indent=2)


def resolve_metric(metric_text: str) -> str:
    m = onto.resolve_metric(metric_text)
    if not m:
        return (f"未在本体中找到指标「{metric_text}」。可用标准指标："
                + "、".join(f"{x['name']}({x['code']})" for x in onto.list_metrics())
                + "。请改用标准指标，禁止自造口径。")
    expr = m.get("expression") or m.get("formula")
    out = {
        "code": m.get("code"),
        "name": m["name"],
        "口径": expr,
        "unit": m.get("unit", ""),
        "可用维度": m.get("dims", []),
        "说明": m.get("note", ""),
    }
    return json.dumps(_for_llm(out, _sql_pairs()), ensure_ascii=False, indent=2)
