"""会话存储：内存为主，落盘 JSON 快照供回看。

一次打开页面直到刷新或关闭是一个会话。每一问的模型上下文是新的，
融合技能留在 skills/_session/<session_id>/，不把上一问的结论带回模型。
"""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from ..config import RUNTIME_DIR


class Session:
    def __init__(self, question: str, model: str, ontology_enabled: bool = True):
        self.session_id = uuid.uuid4().hex[:12]
        self.question = question
        self.model = model
        self.ontology_enabled = ontology_enabled
        self.created = time.time()
        self.events: list[dict] = []
        self.plan: dict | None = None
        self.sub_results: list[dict] = []
        self.final: dict | None = None
        self.tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.turn_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.pending: str | None = None
        self.reply: dict | None = None
        self.ask: dict | None = None

    def begin_question(self, question: str, model: str, ontology_enabled: bool) -> None:
        """同一会话的下一问。清掉这一问的过程，保留已经写成文件的融合专家。"""
        self.question = question
        self.model = model
        self.ontology_enabled = ontology_enabled
        self.events = []
        self.plan = None
        self.sub_results = []
        self.final = None
        self.turn_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.pending = "question"
        self.reply = None
        self.ask = None
        self._persist()

    def add_event(self, ev: dict) -> None:
        ev = {"t": time.time(), **ev}
        self.events.append(ev)
        self._persist()

    def _persist(self) -> None:
        path = RUNTIME_DIR / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "question": self.question,
            "model": self.model,
            "ontology_enabled": self.ontology_enabled,
            "created": self.created,
            "plan": self.plan,
            "events": self.events,
            "sub_results": self.sub_results,
            "final": self.final,
            "tokens": self.tokens,
            "turn_tokens": self.turn_tokens,
            "ask": self.ask,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def set_plan(self, plan: dict) -> None:
        self.plan = plan
        self._persist()

    def add_sub_result(self, r: dict) -> None:
        self.sub_results.append(r)
        self._persist()

    def set_final(self, final: dict) -> None:
        self.final = final
        self._persist()

    def set_ask(self, ask: dict | None) -> None:
        self.ask = ask
        self._persist()

    def add_tokens(self, usage: dict) -> None:
        for k in ("prompt", "completion", "total"):
            amount = usage.get(k, 0) or 0
            self.tokens[k] = self.tokens.get(k, 0) + amount
            self.turn_tokens[k] = self.turn_tokens.get(k, 0) + amount
        self._persist()

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "model": self.model,
            "ontology_enabled": self.ontology_enabled,
            "plan": self.plan,
            "events": self.events,
            "sub_results": self.sub_results,
            "final": self.final,
            "tokens": self.tokens,
            "turn_tokens": self.turn_tokens,
            "ask": self.ask,
        }


_STORE: dict[str, Session] = {}


def create_session(question: str, model: str, ontology_enabled: bool = True) -> Session:
    s = Session(question, model, ontology_enabled)
    s.pending = "question"
    _STORE[s.session_id] = s
    return s


def get_session(session_id: str) -> Session | None:
    return _STORE.get(session_id)


def drop_session(session_id: str) -> None:
    _STORE.pop(session_id, None)
    path = RUNTIME_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()


def live_session_ids() -> set[str]:
    return set(_STORE)
