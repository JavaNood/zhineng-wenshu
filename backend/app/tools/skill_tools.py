"""Skill 工具：list_skills 供主 Agent 选择；skill_card 把 SKILL.md 渲染给子 Agent。"""
from __future__ import annotations
import json
from ..skills import loader as skilllib


def list_skills(session_id: str | None = None) -> str:
    items = [{"id": s["id"], "name": s["name"], "trigger": s["trigger"],
              "description": s["description"]} for s in skilllib.list_skills(session_id)]
    return json.dumps(items, ensure_ascii=False, indent=2)


def skill_card(skill_id: str, session_id: str | None = None) -> str:
    card = skilllib.get_skill(skill_id, session_id)
    if not card:
        return f"未找到 Skill {skill_id}，可用：{[s['id'] for s in skilllib.list_skills(session_id)]}"
    return render_skill_card(card)


def render_skill_card(card: dict) -> str:
    """把 frontmatter 元数据 + Instructions 正文拼成给子 Agent 的完整技能卡。"""
    parts = [
        f"【技能】{card.get('title', card['name'])}（{card['id']}）",
        f"角色：{card.get('expert_role', '')}",
        f"适用触发：{card.get('trigger', '')}",
        f"输入：{', '.join(card.get('inputs', []))}",
        "",
        "## 操作指引（Instructions）",
        card.get("instructions", ""),
        "",
        f"产出：{card.get('output_schema', '')}",
        f"校验：{card.get('validation', '')}",
    ]
    return "\n".join(parts)
