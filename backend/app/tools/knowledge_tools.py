"""本体层与业务语义层工具。关闭本体时调用方不得进入这里。"""
from __future__ import annotations
import json

from ..ontology import loader as onto
from ..semantic import loader as sem


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def read_entity(name: str = "") -> str:
    return _dump(onto.read_entity(name))


def read_value_domain(name: str = "") -> str:
    return _dump(onto.read_value_domain(name))


def read_relations(entity: str = "") -> str:
    return _dump(onto.read_relations(entity))


def read_policies(topic: str = "") -> str:
    return _dump(onto.read_policies(topic))


def read_logical_dataset(name: str = "") -> str:
    return _dump(sem.read_logical_dataset(name))


def resolve_certified_metric(metric: str = "") -> str:
    m = sem.resolve_certified_metric(metric)
    if not m:
        names = "、".join(f"{x['name']}({x['code']})" for x in sem.list_metrics())
        return f"未在认证指标中找到「{metric}」。可用指标：{names}。请改用这些指标，不要自造口径。"
    return _dump({
        "code": m.get("code"),
        "name": m.get("name"),
        "kind": m.get("kind"),
        "口径": m.get("expression") or m.get("formula"),
        "unit": m.get("unit", ""),
        "dataset": m.get("dataset", ""),
        "dims": m.get("dims", []),
        "说明": m.get("note", ""),
        "policy_refs": m.get("policy_refs", []),
    })


def read_sql_plan(metric: str = "") -> str:
    data = sem.read_sql_plan(metric)
    # 规划约束可以带上指标 code，不把「请直接执行的 SQL」放进来。
    plan_view = []
    for p in data.get("plans") or []:
        plan_view.append({k: v for k, v in p.items()})
    metric_view = None
    m = data.get("metric")
    if isinstance(m, dict):
        metric_view = {"code": m.get("code"), "name": m.get("name"), "口径": m.get("expression") or m.get("formula")}
    return _dump({"query": data.get("query", metric), "metric": metric_view, "plans": plan_view})
