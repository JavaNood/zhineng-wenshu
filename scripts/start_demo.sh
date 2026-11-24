#!/usr/bin/env bash
# 一键启动「智能问数 Demo」：后端 8000 + Cloudflare 临时隧道，打印公网链接。
# 用法:
#   bash scripts/start_demo.sh          # 直接启动（使用现有 dist）
#   bash scripts/start_demo.sh --build  # 改动过前端代码时，先重新构建
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT=8000
LOG_DIR="$ROOT/runtime"
mkdir -p "$LOG_DIR"

# 可选：重新构建前端（dist 由后端静态托管，单端口 8000 对外）
if [[ "${1:-}" == "--build" ]]; then
  echo "==> 构建前端 dist ..."
  npm --prefix frontend run build
fi

# 清理旧的后端 / 隧道进程
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
sleep 1

# 启动后端（前台进程已 nohup 后台化，关闭终端不影响）
echo "==> 启动后端 :$PORT"
PYTHONPATH=backend nohup ./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
  > "$LOG_DIR/uvicorn.log" 2>&1 &
echo $! > "$LOG_DIR/backend.pid"

# 等待后端就绪
for _ in $(seq 1 20); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/api/models" && break
  sleep 1
done

# 启动 Cloudflare 临时隧道（免账号、随机域名，关闭即失效）
echo "==> 启动 Cloudflare 临时隧道 ..."
: > "$LOG_DIR/cloudflared.log"
nohup "$LOG_DIR/cloudflared" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
  > "$LOG_DIR/cloudflared.log" 2>&1 &
echo $! > "$LOG_DIR/cf.pid"

# 从日志抓取公网地址
URL=""
for _ in $(seq 1 30); do
  URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/cloudflared.log" | head -1 || true)"
  [[ -n "$URL" ]] && break
  sleep 1
done

cat <<EOF

============================================================
  智能问数 Demo 已启动
  本地预览:  http://127.0.0.1:$PORT
  公网链接:  ${URL:-（隧道还在建立，稍后看 runtime/cloudflared.log 里的 https://...trycloudflare.com）}
============================================================
  停止:  bash scripts/stop_demo.sh
  说明: 关闭终端不影响运行；重开隧道公网域名会变，需重新发链接。
EOF
