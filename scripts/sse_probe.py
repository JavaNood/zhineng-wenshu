"""临时：消费 SSE 事件序列，确诊 consensus/final 是否发出及内容长度。"""
import json, urllib.request, sys

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "2025、2024年的销售额分别是多少，有什么变化，变化的原因是什么？"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/chat",
    data=json.dumps({"question": QUESTION, "model": MODEL}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
start = json.load(urllib.request.urlopen(req, timeout=30))
sid = start["session_id"]
print("session", sid)

events = []
stream = urllib.request.urlopen(f"http://127.0.0.1:8000/api/stream/{sid}", timeout=300)
buf = b""
for chunk in iter(lambda: stream.readline(), b""):
    if chunk == b"\n":
        if buf:
            line = buf.decode("utf-8", "ignore").strip()
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                events.append(ev["type"])
                if ev["type"] == "consensus":
                    print("[consensus] 条数:", len(ev.get("consensus", [])),
                          "分歧:", len(ev.get("divergences", [])))
                elif ev["type"] == "final":
                    print("[final] summary 长度:", len(ev.get("summary", "") or ""),
                          "evidences:", len(ev.get("evidences", [])))
                    print("  summary 前200:", (ev.get("summary") or "")[:200])
                elif ev["type"] == "error":
                    print("[ERROR]", ev.get("message"))
        buf = b""
    else:
        buf += chunk

print("=== 事件序列 ===")
print(" -> ".join(events))
