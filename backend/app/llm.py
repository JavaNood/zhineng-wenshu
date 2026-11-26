"""LLM 封装：DeepSeek OpenAI 兼容，流式 + tokens 累加 + 思维链提取。"""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
from openai import AsyncOpenAI

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEFAULT_MODEL

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY（.env），无法调用 LLM。")
        _client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


def has_key() -> bool:
    return bool(DEEPSEEK_API_KEY)


def _absorb_tool_calls(acc: dict[int, dict], delta) -> None:
    calls = getattr(delta, "tool_calls", None)
    if not calls:
        return
    for tc in calls:
        idx = tc.index if getattr(tc, "index", None) is not None else 0
        slot = acc.setdefault(idx, {
            "id": "", "type": "function", "function": {"name": "", "arguments": ""},
        })
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        if getattr(tc, "type", None):
            slot["type"] = tc.type
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        if getattr(fn, "name", None):
            slot["function"]["name"] += fn.name
        if getattr(fn, "arguments", None):
            slot["function"]["arguments"] += fn.arguments


def _finish_tool_calls(acc: dict[int, dict]) -> list[dict]:
    calls = []
    for idx in sorted(acc):
        slot = acc[idx]
        if not slot["id"]:
            slot["id"] = f"call_{idx}"
        if not slot["function"]["arguments"]:
            slot["function"]["arguments"] = "{}"
        calls.append(slot)
    return calls


async def chat_stream(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    on_content: Callable[[str], Any] | None = None,
    on_reasoning: Callable[[str], Any] | None = None,
    tools: list[dict] | None = None,
    tool_choice: dict | str | None = None,
) -> dict:
    """流式对话。on_content/on_reasoning 为增量回调（用于 SSE 实时推送）。
    返回 {content, reasoning, tool_calls, usage:{prompt,completion,total}}。
    传入 tools 时，工具调用在 tool_calls，不在 content。"""
    client = get_client()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_acc: dict[int, dict] = {}
    usage = {"prompt": 0, "completion": 0, "total": 0}

    # deepseek-reasoner 不接受 temperature，传了会报错；思维链在 reasoning_content。
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    if "reasoner" not in (model or ""):
        kwargs["temperature"] = 0.3
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        # usage 可能挂在带空 choice 的收尾 chunk 上，先读，不能只在 choices 为空时读
        u = getattr(chunk, "usage", None)
        if u:
            usage = {"prompt": u.prompt_tokens or 0,
                     "completion": u.completion_tokens or 0,
                     "total": u.total_tokens or 0}
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning_content", None):
            reasoning_parts.append(delta.reasoning_content)
            if on_reasoning:
                await on_reasoning(delta.reasoning_content)
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            if on_content:
                await on_content(delta.content)
        _absorb_tool_calls(tool_acc, delta)

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": _finish_tool_calls(tool_acc),
        "usage": usage,
    }


async def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    on_reasoning: Callable[[str], Any] | None = None,
) -> dict:
    """需要完整 JSON 的环节。思考模型的 reasoning_content 通过 on_reasoning 原样流出。"""
    return await chat_stream(messages, model=model, on_reasoning=on_reasoning)
