"""本体工具：read_ontology / resolve_metric。"""
from __future__ import annotations
import json
from ..ontology import loader as onto


def read_ontology(topic: str = "") -> str:
    data = onto.read_ontology(topic)
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
    return json.dumps(out, ensure_ascii=False, indent=2)
