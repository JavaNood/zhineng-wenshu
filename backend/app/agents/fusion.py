"""回答结束后的融合。一次函数调用产出技能卡，父专家和祖先由代码写入文件。"""
from __future__ import annotations

from .. import llm
from ..skills import files as skillfiles
from ..skills import loader as skilllib
from ..tools.schemas import assistant_message, calls_to_pending, tool_message

_FIELDS = (
    "title", "description", "trigger", "expert_role",
    "instructions", "output_schema", "validation",
)

_PROPOSE = {
    "type": "function",
    "function": {
        "name": "propose_skill",
        "description": "提交融合后的分析专家技能卡。",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "专家标题，包含职责和称呼"},
                "description": {"type": "string", "description": "擅长什么、何时使用、何时不用"},
                "trigger": {"type": "string"},
                "expert_role": {"type": "string"},
                "instructions": {
                    "type": "string",
                    "description": "Markdown 分析步骤。先沿用一条主线，另一条作为补充校验。不要把两份步骤原样接在一起。步骤控制在 6 步以内，使用认证指标和已有表列。",
                },
                "output_schema": {"type": "string"},
                "validation": {"type": "string"},
            },
            "required": list(_FIELDS),
        },
    },
}

_FUSE_SYS = """你负责把若干分析专家融成一位新的分析专家。
调用 propose_skill 提交技能卡。不要把技能卡写在正文里。
新专家和现有专家是同一结构：何时使用、分析步骤、产出、校验。
步骤是一条可执行的顺序：先沿用其中一条主线，其余专家的检查作为补充。不要把几份步骤原样拼接。
步骤里使用调用方给出的指标和表列，不要发明表或口径。
父专家由系统写入，不要在参数里指定父专家。
"""


def build_fuse_ask(session) -> dict | None:
    subs = [
        item for item in (session.sub_results or [])
        if item.get("skill_id") and item.get("skill_id") != "query"
    ]
    if len(subs) < 2:
        return None
    options = []
    for index, sub in enumerate(subs):
        options.append({
            "key": chr(ord("A") + index),
            "kind": "expert",
            "skill_id": sub["skill_id"],
            "label": sub.get("skill_name") or sub["skill_id"],
        })
    options.append({
        "key": chr(ord("A") + len(subs)),
        "kind": "text",
        "label": "自定义",
    })
    return {
        "type": "ask",
        "card": "fuse",
        "prompt": "勾选你认为回答更优的多位专家，将他们的专家知识融合为 1 个新的专家知识，下次优先让他进行分析。",
        "options": options,
    }


def validate_fuse(selected: list[str], text: str, options: list[dict]) -> str | None:
    expert_ids = [item["skill_id"] for item in options if item.get("kind") == "expert"]
    chosen = []
    for sid in selected:
        if sid in expert_ids and sid not in chosen:
            chosen.append(sid)
    note = (text or "").strip()
    if chosen and note:
        return "选择专家和自己补充不能同时使用，请重新选择"
    if note:
        return None
    if len(chosen) >= 2:
        return None
    return "至少选择 2 位专家，才可进行融合，请重新选择。"


def validate_rename(selected: list[str], text: str, options: list[dict]) -> str | None:
    choice_keys = [item["key"] for item in options if item.get("kind") == "choice"]
    text_keys = [item["key"] for item in options if item.get("kind") == "text"]
    picked = [key for key in selected if key in choice_keys or key in text_keys]
    note = (text or "").strip()
    picked_choice = [key for key in picked if key in choice_keys]
    picked_text = [key for key in picked if key in text_keys]
    if picked_choice and (picked_text or note):
        return "保留默认标题和自定义不能同时使用，请重新选择"
    if picked_choice:
        return None
    if picked_text or note:
        if not note:
            return "需要填写名字"
        return None
    return "请选择保留默认标题，或填写自定义名字"


def _parents(ask: dict, selected: list[str], text: str) -> list[str]:
    experts = [item["skill_id"] for item in ask.get("options") or [] if item.get("kind") == "expert"]
    if (text or "").strip():
        return experts
    chosen = []
    for sid in selected:
        if sid in experts and sid not in chosen:
            chosen.append(sid)
    return chosen


def _close_ancestors(parent_ids: list[str], session_id: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for parent_id in parent_ids:
        if parent_id not in seen:
            ordered.append(parent_id)
            seen.add(parent_id)
        card = skilllib.get_skill(parent_id, session_id) or {}
        for ancestor_id in card.get("ancestors") or []:
            if ancestor_id not in seen:
                ordered.append(ancestor_id)
                seen.add(ancestor_id)
    return ordered


def _materials(session, parent_ids: list[str], note: str) -> str:
    by_id = {item.get("skill_id"): item for item in session.sub_results or []}
    blocks = [f"用户问题：{session.question}"]
    for sid in parent_ids:
        card = skilllib.get_skill(sid, session.session_id) or {}
        sub = by_id.get(sid) or {}
        blocks.append(
            f"## {card.get('title') or sid}（{sid}）\n"
            f"{card.get('instructions') or ''}\n\n"
            f"本轮结论：\n{sub.get('conclusion') or ''}"
        )
    if note:
        blocks.append("用户补充：\n" + note)
    return "\n\n".join(blocks)


def _proposal(tool_calls: list[dict]) -> dict | None:
    for tool in calls_to_pending(tool_calls):
        if tool.get("tool") == "propose_skill":
            return tool
    return None


def _missing(proposal: dict) -> list[str]:
    return [key for key in _FIELDS if not str(proposal.get(key) or "").strip()]


async def run_fusion(session, emit) -> None:
    reply = session.reply or {}
    ask = session.ask or {}
    note = str(reply.get("text") or "").strip()
    parents = _parents(ask, list(reply.get("selected") or []), note)
    if len(parents) < 2 and not note:
        session.set_ask(None)
        await emit({"type": "error", "message": "融合没有完成：没有足够的专家材料。"})
        return
    if note and len(parents) < 1:
        session.set_ask(None)
        await emit({"type": "error", "message": "融合没有完成：本轮没有可融合的专家。"})
        return
    await emit({"type": "ask_status", "text": "正在调用 skills 构建新的专家，请等待..."})
    messages = [
        {"role": "system", "content": _FUSE_SYS},
        {"role": "user", "content": _materials(session, parents, note)},
    ]
    proposal = None
    for _ in range(2):
        resp = await llm.chat_stream(messages, model=session.model, tools=[_PROPOSE])
        session.add_tokens(resp.get("usage") or {})
        proposal = _proposal(resp.get("tool_calls") or [])
        if proposal and not _missing(proposal):
            break
        messages.append(assistant_message(
            resp.get("content") or "",
            resp.get("reasoning") or "",
            resp.get("tool_calls") or [],
        ))
        calls = resp.get("tool_calls") or []
        if calls:
            for call in calls:
                messages.append(tool_message(
                    str(call.get("id") or ""),
                    "字段不完整。请重新调用 propose_skill，title、description、trigger、expert_role、instructions、output_schema、validation 都要有内容。",
                ))
        else:
            messages.append({
                "role": "user",
                "content": "没有收到 propose_skill。请调用这个函数提交技能卡，不要把技能卡写在正文里。",
            })
        proposal = None
    if not proposal or _missing(proposal):
        session.set_ask(None)
        await emit({"type": "error", "message": "融合没有完成：模型没有提交完整的技能卡。"})
        await emit({"type": "tokens", "tokens": session.turn_tokens})
        return
    ancestors = _close_ancestors(parents, session.session_id)
    fuse_id = skillfiles.next_fuse_id(session.session_id)
    skillfiles.write_session_skill(
        session.session_id,
        fuse_id,
        title=str(proposal["title"]).strip(),
        description=str(proposal["description"]).strip(),
        trigger=str(proposal["trigger"]).strip(),
        expert_role=str(proposal["expert_role"]).strip(),
        instructions=str(proposal["instructions"]).strip(),
        output_schema=str(proposal["output_schema"]).strip(),
        validation=str(proposal["validation"]).strip(),
        parents=parents,
        ancestors=ancestors,
    )
    title = str(proposal["title"]).strip()
    session.set_ask({
        "type": "ask",
        "card": "rename",
        "skill_id": fuse_id,
        "prompt": "专家已经构建完成，你可以为专家做一个新的命名。",
        "options": [
            {"key": "A", "kind": "choice", "label": title},
            {"key": "B", "kind": "text", "label": "自定义"},
        ],
    })
    await emit({"type": "tokens", "tokens": session.turn_tokens})
    await emit(session.ask)


async def run_rename(session, emit) -> None:
    reply = session.reply or {}
    ask = session.ask or {}
    fuse_id = str(ask.get("skill_id") or "")
    note = str(reply.get("text") or "").strip()
    selected = list(reply.get("selected") or [])
    text_keys = [item["key"] for item in ask.get("options") or [] if item.get("kind") == "text"]
    if note and any(key in text_keys for key in selected):
        skillfiles.set_title(session.session_id, fuse_id, note)
    session.set_ask(None)
    card = skilllib.get_skill(fuse_id, session.session_id) or {}
    await emit({
        "type": "ask_closed",
        "card": "rename",
        "skill_id": fuse_id,
        "title": card.get("title") or fuse_id,
    })


def reply_error(session, card: str, action: str, selected: list[str], text: str) -> str | None:
    ask = session.ask or {}
    if ask.get("card") != card:
        return "没有待回答的问题"
    if action == "cancel":
        return None
    if action != "confirm":
        return "无法识别的操作"
    options = ask.get("options") or []
    if card == "fuse":
        return validate_fuse(selected, text, options)
    if card == "rename":
        return validate_rename(selected, text, options)
    return "无法识别的问题"
