"""技能库加载：读取 skills/<name>/SKILL.md（官方 Skill 形态）。

SKILL.md = YAML frontmatter（name/description/trigger/inputs/...）+ Markdown 正文（Instructions）。
frontmatter 始终用于"发现与选择"，正文在被选中后注入子 Agent。
"""
from __future__ import annotations
import yaml
from functools import lru_cache
from pathlib import Path
from ..config import SKILL_DIR


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """切分 '---\\n...---\\n正文'，返回 (frontmatter dict, 正文 markdown)。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return meta, body


@lru_cache(maxsize=1)
def load_skills() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(SKILL_DIR.glob("*/SKILL.md")):
        raw = p.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name") or p.parent.name
        card = {
            "id": name,
            "dir": p.parent.name,
            "name": meta.get("name", name),
            "title": meta.get("title", meta.get("name", name)),
            "description": meta.get("description", ""),
            "trigger": meta.get("trigger", ""),
            "expert_role": meta.get("expert_role", ""),
            "inputs": meta.get("inputs", []),
            "output_schema": meta.get("output_schema", ""),
            "validation": meta.get("validation", ""),
            "instructions": body,
            "path": str(p),
        }
        out[name] = card
    return out


def list_skills() -> list[dict]:
    return [
        {"id": c["id"], "name": c["title"], "expert_role": c.get("expert_role", ""),
         "trigger": c.get("trigger", ""), "description": c.get("description", "")}
        for c in load_skills().values()
    ]


def get_skill(skill_id: str) -> dict | None:
    return load_skills().get(skill_id)
