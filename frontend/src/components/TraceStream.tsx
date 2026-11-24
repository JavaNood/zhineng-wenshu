import { useEffect, useRef, useState } from "react";
import type { TimelineNode } from "../hooks/useQa";
import { Md } from "./Md";

const TOOL_ICON: Record<string, string> = {
  resolve_metric: "📚",
  read_ontology: "📚",
  list_skills: "🧭",
  run_sql: "⚙️",
  report: "📝",
  dispatch: "👥",
};

type ToolNode = Extract<TimelineNode, { kind: "tool" }>;
type ThinkNode = Extract<TimelineNode, { kind: "think" }>;

function ThinkCard({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  const wasLive = useRef(live);
  useEffect(() => {
    if (wasLive.current && !live) setOpen(false);
    wasLive.current = live;
  }, [live]);
  const preview = text.trim().split("\n").map((s) => s.trim()).find(Boolean) || "思考";
  return (
    <div className="react-think">
      <button type="button" className="think-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="react-think-tag">思考</span>
        {!open && <span className="think-preview">{preview}</span>}
        <span className="think-caret">{open ? "收起" : "展开"}</span>
      </button>
      {open && <div className="think-body">{text}</div>}
    </div>
  );
}

function ToolBar({ item }: { item: ToolNode }) {
  const [showBody, setShowBody] = useState(false);
  const [showSql, setShowSql] = useState(false);
  const icon = TOOL_ICON[item.tool] || "🔧";
  const hasBody = !!item.body;
  const hasSql = !!item.sql;
  return (
    <div className="trace-tool">
      <span className="trace-tool-icon">{icon}</span>
      <div className="trace-tool-body">
        <div className="trace-tool-title">
          <span className="tool-kind">工具</span>
          {item.tool}
        </div>
        {item.detail && <div className="trace-tool-detail">{item.detail}</div>}
        {hasBody && (
          <>
            <button type="button" className="trace-sql-toggle" onClick={() => setShowBody((s) => !s)}>
              {showBody ? "收起结果" : "查看结果"}
            </button>
            {showBody && <pre className="trace-body">{item.body}</pre>}
          </>
        )}
        {hasSql && (
          <>
            <button type="button" className="trace-sql-toggle" onClick={() => setShowSql((s) => !s)}>
              {showSql ? "收起 SQL" : "查看 SQL"}
            </button>
            {showSql && <pre className="trace-sql">{item.sql}</pre>}
          </>
        )}
      </div>
    </div>
  );
}

type Block =
  | { t: "think"; node: ThinkNode }
  | { t: "act"; node: Extract<TimelineNode, { kind: "act" }> }
  | { t: "answer"; node: Extract<TimelineNode, { kind: "answer" }> }
  | { t: "tool"; node: ToolNode };

/** 主 Agent 时间线：这一步要做什么、可折叠的思维链、单独的工具结果，按到达顺序穿插。 */
export function TraceStream({ items }: { items: TimelineNode[] }) {
  if (items.length === 0) return null;
  const blocks: Block[] = items.map((it) => {
    if (it.kind === "think") return { t: "think", node: it };
    if (it.kind === "act") return { t: "act", node: it };
    if (it.kind === "answer") return { t: "answer", node: it };
    return { t: "tool", node: it };
  });

  return (
    <div className="react-stream">
      {blocks.map((b, i) => {
        if (b.t === "think") {
          return <ThinkCard key={i} text={b.node.text} live={i === blocks.length - 1} />;
        }
        if (b.t === "act") {
          return <div key={i} className="react-act">{b.node.text}</div>;
        }
        if (b.t === "answer") {
          return (
            <div key={i} className="main-answer">
              <Md text={b.node.text} />
            </div>
          );
        }
        return <ToolBar key={i} item={b.node} />;
      })}
    </div>
  );
}
