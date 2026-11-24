"""FastAPI 入口：REST 发起对话 + SSE 事件流 + 会话快照 + 模型列表。"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import resolve_model, MODEL_WHITELIST, has_key
from .db import init_db
from .store.session import create_session, get_session
from .agents.supervisor import orchestrate

app = FastAPI(title="智能问数 Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)


class ChatRequest(BaseModel):
    question: str
    model: str | None = None


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/api/models")
def models():
    return {"models": MODEL_WHITELIST, "default": "deepseek-chat",
            "key_configured": has_key()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    model = resolve_model(req.model)
    s = create_session(req.question.strip(), model)
    return {"session_id": s.session_id, "model": model, "key_configured": has_key()}


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
            await orchestrate(s.question, s.model, emit, s)
        except Exception as e:  # 不降级：推错误事件，由前端展示
            await q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await q.put(None)

    asyncio.create_task(runner())

    async def gen():
        yield "retry: 2000\n\n"
        while True:
            ev = await q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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

