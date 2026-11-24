"""本地真实复现：直接调 orchestrate，打印每个事件与异常栈（不经过前端）。"""
import asyncio
import sys
import traceback

from app.agents.supervisor import orchestrate
from app.store.session import create_session


async def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "2025年的销售额是多少"
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"
    s = create_session(question, model)

    async def emit(ev: dict):
        t = ev.get("type")
        if t in ("reasoning",):
            return  # 不刷屏
        print(f"[{t}] {str(ev)[:300]}")

    try:
        await orchestrate(s.question, s.model, emit, s)
    except Exception:
        traceback.print_exc()
    print("=== tokens:", s.tokens)
    print("=== sub_results:", len(s.sub_results))
    print("=== final:", (s.final or {}).get("summary", "")[:300])


if __name__ == "__main__":
    asyncio.run(main())
