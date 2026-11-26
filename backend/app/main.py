"""FastAPI 入口：REST 发起对话 + SSE 事件流 + 会话快照 + 模型列表。"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import resolve_model, MODEL_WHITELIST, DEFAULT_MODEL, has_key
from .db import init_db
from .store.session import create_session, drop_session, get_session, live_session_ids
from .agents.supervisor import orchestrate
from .agents.fusion import build_fuse_ask, reply_error, run_fusion, run_rename
from .skills.files import delete_session_dir, sweep_session_dirs
from .skills.loader import get_skill

app = FastAPI(title="智能问数 Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)


class ChatRequest(BaseModel):
    question: str
    model: str | None = None
    ontology_enabled: bool = True
    session_id: str | None = None


class ReplyRequest(BaseModel):
    card: str
    action: str
    selected: list[str] = []
    text: str = ""


@app.on_event("startup")
def _startup():
    init_db()
    sweep_session_dirs(live_session_ids())
    from .ontology.loader import validate_ontology
    problems = validate_ontology()
    if problems:
        print("知识校验未通过:", problems)


@app.get("/api/models")
def models():
    return {"models": MODEL_WHITELIST, "default": DEFAULT_MODEL,
            "key_configured": has_key()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    model = resolve_model(req.model)
    question = req.question.strip()
    if req.session_id:
        s = get_session(req.session_id)
        if not s:
            return JSONResponse({"error": "session not found"}, status_code=404)
        s.begin_question(question, model, req.ontology_enabled)
    else:
        s = create_session(question, model, req.ontology_enabled)
    return {"session_id": s.session_id, "model": model, "key_configured": has_key()}


@app.post("/api/session/{session_id}/reply")
def reply(session_id: str, req: ReplyRequest):
    s = get_session(session_id)
    if not s:
        return JSONResponse({"error": "session not found"}, status_code=404)
    err = reply_error(s, req.card, req.action, req.selected, req.text)
    if err:
        return {"accepted": False, "error": err}
    if req.action == "cancel":
        s.reply = None
        s.pending = None
        s.set_ask(None)
        return {"accepted": True, "done": True}
    s.reply = {
        "card": req.card,
        "action": req.action,
        "selected": list(req.selected),
        "text": req.text,
    }
    s.pending = "reply"
    return {"accepted": True, "done": False}


@app.post("/api/session/{session_id}/end")
def end_session(session_id: str):
    delete_session_dir(session_id)
    drop_session(session_id)
    return {"ok": True}


@app.get("/api/stream/{session_id}")
async def stream(session_id: str):
    s = get_session(session_id)
    if not s:
        return {"error": "session not found"}

    q: asyncio.Queue = asyncio.Queue()

    async def emit(ev: dict):
        await q.put(ev)

    async def runner():
        try:
            job = s.pending
            s.pending = None
            if job == "reply":
                card = (s.ask or {}).get("card")
                if card == "fuse":
                    await run_fusion(s, emit)
                elif card == "rename":
                    await run_rename(s, emit)
                else:
                    await emit({"type": "error", "message": "没有待回答的问题"})
            elif job == "question":
                await orchestrate(s.question, s.model, emit, s)
                ask = build_fuse_ask(s)
                if ask:
                    s.set_ask(ask)
                    await emit(ask)
            # 流结束后浏览器可能重连。没有新任务时直接结束，不再跑一遍。
        except Exception as e:  # 不降级：推错误事件，由前端展示
            await q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await q.put(None)

    task = asyncio.create_task(runner())

    # 临时隧道会把很小的 SSE 包攒到连接结束才下发。每条事件后垫一段注释，迫使它马上转发。
    pad = ":" + (" " * 8192) + "\n\n"

    async def gen():
        try:
            yield "retry: 2000\n\n"
            yield pad
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                yield pad
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/session/{session_id}/skill/{skill_id}")
def session_skill(session_id: str, skill_id: str):
    s = get_session(session_id)
    if not s or "/" in skill_id or ".." in skill_id:
        return JSONResponse({"error": "skill not found"}, status_code=404)
    card = get_skill(skill_id, session_id)
    if not card or card.get("scope") != "session" or card.get("session_id") != session_id:
        return JSONResponse({"error": "skill not found"}, status_code=404)
    text = Path(card["path"]).read_text(encoding="utf-8")
    return {"id": card["id"], "title": card.get("title") or card["id"], "text": text}


@app.get("/api/session/{session_id}")
def session_snapshot(session_id: str):
    s = get_session(session_id)
    if not s:
        return {"error": "session not found"}
    return s.snapshot()


# —— 前端静态托管：build 产物 frontend/dist，单端口对外，隧道只指 8000 ——
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True))

