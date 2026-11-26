"""会话融合技能的文件。只读写 skills/_session/<session_id>/，不进基础技能缓存。"""
from __future__ import annotations
import shutil
from pathlib import Path

import yaml

from ..config import SKILL_DIR
from .loader import parse_frontmatter

SESSION_ROOT_NAME = "_session"


def session_dir(session_id: str) -> Path:
    return SKILL_DIR / SESSION_ROOT_NAME / session_id


def render_skill(meta: dict, body: str) -> str:
    dumped = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    text = (body or "").strip()
    return f"---\n{dumped}\n---\n\n{text}\n"


def next_fuse_id(session_id: str) -> str:
    root = session_dir(session_id)
    numbers = []
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith("fuse-"):
                continue
            suffix = child.name.removeprefix("fuse-")
            if suffix.isdigit():
                numbers.append(int(suffix))
    return f"fuse-{max(numbers, default=0) + 1}"


def write_session_skill(
    session_id: str,
    fuse_id: str,
    *,
    title: str,
    description: str,
    trigger: str,
    expert_role: str,
    instructions: str,
    output_schema: str,
    validation: str,
    parents: list[str],
    ancestors: list[str],
) -> Path:
    root = session_dir(session_id) / fuse_id
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": fuse_id,
        "title": title,
        "description": description,
        "trigger": trigger,
        "expert_role": expert_role,
        "inputs": [],
        "output_schema": output_schema,
        "validation": validation,
        "scope": "session",
        "session_id": session_id,
        "parents": list(parents),
        "ancestors": list(ancestors),
    }
    path = root / "SKILL.md"
    path.write_text(render_skill(meta, instructions), encoding="utf-8")
    return path


def set_title(session_id: str, fuse_id: str, title: str) -> None:
    path = session_dir(session_id) / fuse_id / "SKILL.md"
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    meta["title"] = title.strip()
    path.write_text(render_skill(meta, body), encoding="utf-8")


def delete_session_dir(session_id: str) -> None:
    root = session_dir(session_id)
    if root.exists():
        shutil.rmtree(root)


def sweep_session_dirs(live_ids: set[str]) -> None:
    root = SKILL_DIR / SESSION_ROOT_NAME
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name not in live_ids:
            shutil.rmtree(child, ignore_errors=True)
