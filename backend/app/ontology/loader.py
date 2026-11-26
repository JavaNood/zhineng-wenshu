"""本体层加载：实体、值域、关系、策略。不读取认证指标公式。"""
from __future__ import annotations
import yaml
from functools import lru_cache
from pathlib import Path

from ..config import ONTOLOGY_DIR


def _load(name: str) -> dict:
    path = Path(ONTOLOGY_DIR) / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_entities() -> list[dict]:
    return list(_load("entities.yaml").get("entities") or [])


@lru_cache(maxsize=1)
def load_domains() -> list[dict]:
    return list(_load("value_domains.yaml").get("value_domains") or [])


@lru_cache(maxsize=1)
def load_relations() -> list[dict]:
    return list(_load("relations.yaml").get("relations") or [])


@lru_cache(maxsize=1)
def load_policies() -> list[dict]:
    return list(_load("policies.yaml").get("policies") or [])


def _hit(text: str, *parts: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p and (p == t or p in t or t in p) for p in parts)


def read_entity(name: str = "") -> dict:
    rows = load_entities()
    if not (name or "").strip():
        return {"entities": [{"name": e["name"], "grain": e.get("grain", "")} for e in rows]}
    hit = [e for e in rows if _hit(name, e["name"], e.get("identifier", ""))]
    return {"query": name, "entities": hit}


def read_value_domain(name: str = "") -> dict:
    rows = load_domains()
    if not (name or "").strip():
        return {"domains": [{"name": d["name"], "entity": d.get("entity"), "attribute": d.get("attribute")} for d in rows]}
    hit = [d for d in rows if _hit(name, d["name"], d.get("attribute", ""), f"{d.get('entity', '')}.{d.get('attribute', '')}")]
    return {"query": name, "domains": hit}


def read_relations(entity: str = "") -> dict:
    rows = load_relations()
    if not (entity or "").strip():
        return {"relations": [{"id": r["id"], "from": r["from"], "to": r["to"], "type": r.get("type")} for r in rows]}
    hit = [r for r in rows if entity in r["id"] or entity in r["from"] or entity in r["to"]]
    return {"query": entity, "relations": hit}


def read_policies(topic: str = "") -> dict:
    rows = load_policies()
    if not (topic or "").strip():
        return {"policies": [{"id": p["id"], "summary": p.get("summary", "")} for p in rows]}
    hit = [p for p in rows if _hit(topic, p["id"], p.get("summary", ""), p.get("text", ""))]
    return {"query": topic, "policies": hit}


def validate_ontology() -> list[str]:
    """实体属性、值域、关系、策略，以及语义层引用。返回问题清单。"""
    problems: list[str] = []
    attrs: dict[str, set[str]] = {}
    for ent in load_entities():
        attrs[ent["name"]] = {a["name"] for a in ent.get("attributes") or []}
    known_policies = {p["id"] for p in load_policies()}
    for ent in load_entities():
        for pid in ent.get("policy_refs") or []:
            if pid not in known_policies:
                problems.append(f"实体 {ent['name']} 引用了不存在的策略 {pid}")
    for dom in load_domains():
        ent, attr = dom.get("entity"), dom.get("attribute")
        if ent not in attrs:
            problems.append(f"值域 {dom.get('name')} 的实体不存在：{ent}")
        elif attr and attr not in attrs[ent]:
            problems.append(f"值域 {dom.get('name')} 的属性不存在：{ent}.{attr}")
    known_rel = set()
    for rel in load_relations():
        known_rel.add(rel["id"])
        for end in (rel.get("from"), rel.get("to")):
            ent, _, attr = (end or "").partition(".")
            if ent not in attrs:
                problems.append(f"关系 {rel.get('id')} 的实体不存在：{end}")
            elif attr and attr not in attrs[ent]:
                problems.append(f"关系 {rel.get('id')} 的属性不存在：{end}")
    from ..semantic.loader import validate_semantic
    problems.extend(validate_semantic(known_rel, known_policies))
    return problems
