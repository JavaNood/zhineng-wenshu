// API 封装：模型列表、发起对话、SSE 流式连接
export interface ModelInfo {
  models: string[];
  default: string;
  key_configured: boolean;
}

export interface ChatStart {
  session_id: string;
  model: string;
  key_configured: boolean;
}

export async function fetchModels(): Promise<ModelInfo> {
  const r = await fetch("/api/models");
  return r.json();
}

export async function startChat(
  question: string,
  model: string,
  ontologyEnabled = true,
  sessionId?: string,
): Promise<ChatStart> {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      model,
      ontology_enabled: ontologyEnabled,
      session_id: sessionId || undefined,
    }),
  });
  const data = await r.json();
  if (!r.ok) {
    throw new Error(data.error || "无法开始对话");
  }
  return data;
}

export async function postReply(
  sessionId: string,
  body: { card: string; action: string; selected: string[]; text: string },
): Promise<{ accepted: boolean; done?: boolean; error?: string }> {
  const r = await fetch(`/api/session/${sessionId}/reply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) {
    throw new Error(data.error || "无法提交");
  }
  return data;
}

export type EventHandler = (ev: any) => void;

// 用 EventSource 连接 SSE（/api/stream/{session_id} 为 GET，会话在 POST /api/chat 已建立）
export function connectStream(sessionId: string, onEvent: EventHandler, onDone: () => void) {
  const es = new EventSource(`/api/stream/${sessionId}`);
  es.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data));
    } catch {
      /* 忽略非法帧 */
    }
  };
  es.onerror = () => {
    es.close();
    onDone();
  };
  return () => es.close();
}
