"""技能库加载：读取 skills/<name>/SKILL.md（官方 Skill 形态）。

SKILL.md = YAML frontmatter（name/description/trigger/inputs/...）+ Markdown 正文（Instructions）。
frontmatter 始终用于"发现与选择"，正文在被选中后注入子 Agent。

基础技能只扫 skills 下一层，并且跳过 _session。会话融合技能按 session_id 另读，不进全局缓存。
"""
from __future__ import annotations
import yaml
from functools import lru_cache
from pathlib import Path
from ..config import SKILL_DIR

SESSION_ROOT_NAME = "_session"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """切分 '---\\n...---\\n正文'，返回 (frontmatter dict, 正文 markdown)。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return meta, body


def _card(path: Path, meta: dict, body: str) -> dict:
    name = meta.get("name") or path.parent.name
    return {
        "id": name,
        "dir": path.parent.name,
        "name": meta.get("name", name),
        "title": meta.get("title", meta.get("name", name)),
        "description": meta.get("description", ""),
        "trigger": meta.get("trigger", ""),
        "expert_role": meta.get("expert_role", ""),
        "inputs": meta.get("inputs", []),
        "output_schema": meta.get("output_schema", ""),
        "validation": meta.get("validation", ""),
        "instructions": body,
        "path": str(path),
        "scope": meta.get("scope") or "base",
        "session_id": meta.get("session_id") or "",
        "parents": list(meta.get("parents") or []),
        "ancestors": list(meta.get("ancestors") or []),
    }


def _read_card(path: Path) -> dict | None:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    if not meta:
        return None
    return _card(path, meta, body)


@lru_cache(maxsize=1)
def load_skills() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(SKILL_DIR.glob("*/SKILL.md")):
        if path.parent.name.startswith("_") or path.parent.name == SESSION_ROOT_NAME:
            continue
        card = _read_card(path)
        if not card or card.get("scope") == "session":
            continue
        out[card["id"]] = card
    return out


def load_session_skills(session_id: str) -> dict[str, dict]:
    root = SKILL_DIR / SESSION_ROOT_NAME / session_id
    if not session_id or not root.is_dir():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(root.glob("*/SKILL.md")):
        card = _read_card(path)
        if not card or card.get("scope") != "session":
            continue
        if card.get("session_id") != session_id:
            continue
        out[card["id"]] = card
    return out


def skills_for(session_id: str | None = None) -> dict[str, dict]:
    found = dict(load_skills())
    if session_id:
        found.update(load_session_skills(session_id))
    return found


def _parent_titles(card: dict) -> list[str]:
    titles = []
    session_id = card.get("session_id") or ""
    known = skills_for(session_id)
    for parent_id in card.get("parents") or []:
        parent = known.get(parent_id)
        titles.append((parent or {}).get("title") or parent_id)
    return titles


def catalog_description(card: dict) -> str:
    text = card.get("description") or ""
    if card.get("scope") != "session":
        return text
    titles = _parent_titles(card)
    note = "本会话用户融合的专家"
    if titles:
        note += "，由" + "、".join(titles) + "融合"
    note += "。问题同样适合多位专家时优先考虑。"
    return (text + " " + note).strip()


def list_skills(session_id: str | None = None) -> list[dict]:
    return [
        {
            "id": card["id"],
            "name": card["title"],
            "expert_role": card.get("expert_role", ""),
            "trigger": card.get("trigger", ""),
            "description": catalog_description(card),
            "scope": card.get("scope") or "base",
        }
        for card in skills_for(session_id).values()
    ]


def get_skill(skill_id: str, session_id: str | None = None) -> dict | None:
    if session_id:
        found = load_session_skills(session_id).get(skill_id)
        if found:
            return found
    return load_skills().get(skill_id)
