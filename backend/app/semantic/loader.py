"""业务语义层加载：逻辑数据集、认证指标、SQL 规划约束。"""
from __future__ import annotations
import re
import yaml
from functools import lru_cache
from pathlib import Path

from ..config import SEMANTIC_DIR

_SKIP = {"sum", "avg", "count", "max", "min", "target", "sales", "gross"}


def _load(name: str) -> dict:
    path = Path(SEMANTIC_DIR) / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_datasets() -> list[dict]:
    return list(_load("logical_datasets.yaml").get("logical_datasets") or [])


@lru_cache(maxsize=1)
def load_metrics() -> list[dict]:
    return list(_load("certified_metrics.yaml").get("metrics") or [])


@lru_cache(maxsize=1)
def load_plans() -> list[dict]:
    return list(_load("sql_planner.yaml").get("plans") or [])


def list_metrics() -> list[dict]:
    return [
        {"code": m.get("code"), "name": m["name"], "kind": m.get("kind"),
         "expression": m.get("expression") or m.get("formula"), "unit": m.get("unit", "")}
        for m in load_metrics()
    ]


def _names(m: dict) -> list[str]:
    return [m.get("name") or "", m.get("code") or ""] + list(m.get("synonyms") or [])


def resolve_certified_metric(text: str) -> dict | None:
    t = (text or "").strip()
    if not t:
        return None
    for m in load_metrics():
        if m.get("name") == t or m.get("code") == t:
            return m
        for syn in m.get("synonyms") or []:
            if syn and syn in t:
                return m
    for m in load_metrics():
        if any(n and n in t for n in _names(m)):
            return m
    return None


def read_logical_dataset(name: str = "") -> dict:
    rows = load_datasets()
    if not (name or "").strip():
        return {"datasets": [{"id": d["id"], "entity": d.get("entity"), "file": d.get("file")} for d in rows]}
    hit = [d for d in rows if name in (d["id"], d.get("entity"), d.get("file"), (d.get("file") or "").replace(".csv", ""))]
    return {"query": name, "datasets": hit}


def read_sql_plan(metric: str = "") -> dict:
    rows = load_plans()
    if not (metric or "").strip():
        return {"plans": [{"metric": p.get("metric"), "dataset": p.get("dataset")} for p in rows]}
    m = resolve_certified_metric(metric)
    code = m["code"] if m else metric
    hit = [p for p in rows if p.get("metric") in (code, metric)]
    return {"query": metric, "metric": m, "plans": hit}


def validate_semantic(known_rel: set[str], known_policies: set[str]) -> list[str]:
    problems: list[str] = []
    datasets = {d["id"] for d in load_datasets()}
    defined = {m.get("code") for m in load_metrics() if m.get("kind") == "base"}
    for m in load_metrics():
        if m.get("kind") == "base":
            if not m.get("expression"):
                problems.append(f"基础指标 {m.get('code')} 缺少 expression")
            if m.get("dataset") not in datasets:
                problems.append(f"基础指标 {m.get('code')} 的逻辑数据集不存在")
        else:
            formula = m.get("formula") or ""
            for tok in re.findall(r"[a-z][a-z0-9_]*", formula):
                if tok in _SKIP:
                    continue
                if tok not in defined and tok not in {x.get("code") for x in load_metrics()}:
                    problems.append(f"派生指标 {m.get('code')} 引用未定义指标 {tok}")
        for pid in m.get("policy_refs") or []:
            if pid not in known_policies:
                problems.append(f"指标 {m.get('code')} 引用了不存在的策略 {pid}")
    for p in load_plans():
        if p.get("metric") not in {m.get("code") for m in load_metrics()}:
            problems.append(f"SQL 规划引用了不存在的指标 {p.get('metric')}")
        if p.get("dataset") and p.get("dataset") not in datasets:
            problems.append(f"SQL 规划 {p.get('metric')} 的逻辑数据集不存在")
        for rid in p.get("relations") or []:
            if rid not in known_rel:
                problems.append(f"SQL 规划 {p.get('metric')} 引用了不存在的关系 {rid}")
    return problems
