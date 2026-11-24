#!/usr/bin/env bash
# 停止「智能问数 Demo」：关闭后端与 Cloudflare 隧道。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT/runtime/cf.pid" ]]; then
  kill "$(cat "$ROOT/runtime/cf.pid")" 2>/dev/null || true
fi
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
pkill -f "uvicorn app.main:app" 2>/dev/null || true

echo "已停止后端与隧道。"
