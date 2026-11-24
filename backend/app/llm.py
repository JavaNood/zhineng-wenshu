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


async def chat_stream(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    on_content: Callable[[str], Any] | None = None,
    on_reasoning: Callable[[str], Any] | None = None,
) -> dict:
    """流式对话。on_content/on_reasoning 为增量回调（用于 SSE 实时推送）。
    返回 {content, reasoning, usage:{prompt,completion,total}}。"""
    client = get_client()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = {"prompt": 0, "completion": 0, "total": 0}

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.3,
    )
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

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "usage": usage,
    }


async def chat(messages: list[dict], model: str = DEFAULT_MODEL) -> dict:
    """非流式：用于意图分类、Plan 生成等需要完整 JSON 的环节。"""
    return await chat_stream(messages, model=model)
