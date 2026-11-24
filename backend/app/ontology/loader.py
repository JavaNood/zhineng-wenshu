"""本体加载与查询：read_ontology / resolve_metric 的底层。"""
from __future__ import annotations
import re
import yaml
from functools import lru_cache
from pathlib import Path

from ..config import ONTOLOGY_FILE


@lru_cache(maxsize=1)
def load_ontology(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ONTOLOGY_FILE
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _all_metrics(ont: dict) -> list[dict]:
    """基础指标 + 派生指标合并列表。"""
    rows = []
    for m in ont.get("metrics", []):
        rows.append({**m, "kind": "base"})
    for f in ont.get("metric_formulas", []):
        rows.append({**f, "kind": "derived"})
    return rows


def resolve_metric(text: str) -> dict | None:
    """自然语言指标对齐标准指标：按 name / synonyms / code 命中。"""
    ont = load_ontology()
    t = (text or "").strip()
    for m in _all_metrics(ont):
        if m["name"] == t or m.get("code") == t:
            return m
        for syn in m.get("synonyms", []):
            if syn and syn in t:
                return m
    # 退一步：包含匹配
    for m in _all_metrics(ont):
        names = [m["name"], m.get("code", "")] + m.get("synonyms", [])
        if any(n and n in t for n in names):
            return m
    return None


def list_metrics() -> list[dict]:
    ont = load_ontology()
    return [
        {"code": m.get("code"), "name": m["name"], "kind": m["kind"],
         "expression": m.get("expression") or m.get("formula"),
         "unit": m.get("unit", "")}
        for m in _all_metrics(ont)
    ]


def list_dimensions() -> list[dict]:
    return load_ontology().get("dimensions", [])


def get_relationships() -> list[dict]:
    return load_ontology().get("relationships", [])


def get_dataset(source: str) -> dict | None:
    for ds in load_ontology().get("datasets", []):
        if ds["source"] == source:
            return ds
    return None


def read_ontology(topic: str = "") -> dict:
    """read_ontology 工具底层：按主题返回相关本体片段；空主题返回摘要。"""
    ont = load_ontology()
    if not topic:
        return {
            "meta": ont.get("meta"),
            "instructions": ont.get("instructions"),
            "metrics": [{"code": m.get("code"), "name": m["name"],
                         "expression": m.get("expression") or m.get("formula")}
                        for m in _all_metrics(ont)],
            "datasets": [ds["source"] for ds in ont.get("datasets", [])],
            "dimensions": [d["name"] for d in ont.get("dimensions", [])],
        }
    topic = topic.strip()
    # 指标命中
    hits = [m for m in _all_metrics(ont)
            if m["name"] in topic or m.get("code") in topic
            or any(s in topic for s in m.get("synonyms", []))]
    # 表命中
    tables = [ds for ds in ont.get("datasets", []) if ds["source"] in topic]
    # 维度命中
    dims = [d for d in ont.get("dimensions", [])
            if d["code"] in topic or d["name"] in topic]
    return {
        "topic": topic,
        "metrics": hits,
        "tables": tables,
        "dimensions": dims,
        "relationships": ont.get("relationships", []),
    }


def validate_ontology() -> list[str]:
    """3.3 校验：缺字段 / 关系两端不存在 / 派生引用未定义。返回问题清单（空=通过）。"""
    problems: list[str] = []
    ont = load_ontology()
    defined = {m.get("code") for m in ont.get("metrics", [])}
    # 基础指标必备字段
    for m in ont.get("metrics", []):
        for fld in ("expression", "additivity", "dims"):
            if fld not in m:
                problems.append(f"基础指标 {m.get('code')} 缺字段 {fld}")
    # 派生指标：公式中出现的小写标识符（指标 code）必须已定义；SQL 聚合函数与表别名跳过
    _SKIP = {"sum", "avg", "count", "max", "min", "target", "sales", "gross"}
    for f in ont.get("metric_formulas", []):
        for tok in re.findall(r"[a-z][a-z0-9_]*", f.get("formula") or ""):
            if tok in _SKIP:
                continue
            if tok not in defined:
                problems.append(f"派生指标 {f.get('code')} 引用未定义指标 {tok}")
    # 关系两端表/字段真实存在
    known_fields: dict[str, set[str]] = {}
    for ds in ont.get("datasets", []):
        known_fields[ds["source"]] = {f["name"] for f in ds["fields"]}
    for rel in ont.get("relationships", []):
        for end in (rel["from"], rel["to"]):
            tbl, _, col = end.partition(".")
            if tbl not in known_fields:
                problems.append(f"关系 {end} 表不存在")
            elif col and col not in known_fields[tbl]:
                problems.append(f"关系 {end} 字段不存在")
    return problems
