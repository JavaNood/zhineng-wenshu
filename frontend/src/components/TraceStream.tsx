import { useState } from "react";
import type { TimelineNode } from "../hooks/useQa";
import { Md } from "./Md";

const TOOL_ICON: Record<string, string> = {
  resolve_metric: "📚",
  read_ontology: "📚",
  list_skills: "🧭",
  run_sql: "⚙️",
};

type ToolNode = Extract<TimelineNode, { kind: "tool" }>;

function ToolBar({ item, compact }: { item: ToolNode; compact?: boolean }) {
  const [showBody, setShowBody] = useState(false);
  const [showSql, setShowSql] = useState(false);
  const icon = TOOL_ICON[item.tool] || "🔧";
  const hasBody = !!item.body;
  const hasSql = !!item.sql;
  return (
    <div className={"trace-tool" + (compact ? " trace-tool--compact" : "")}>
      <span className="trace-tool-icon">{icon}</span>
      <div className="trace-tool-body">
        <div className="trace-tool-title">{item.title}</div>
        {item.detail && <div className="trace-tool-detail">{item.detail}</div>}
        {hasBody && (
          <>
            <button className="trace-sql-toggle" onClick={() => setShowBody((s) => !s)}>
              {showBody ? "收起内容" : "查看内容"}
            </button>
            {showBody && <pre className="trace-body">{item.body}</pre>}
          </>
        )}
        {hasSql && (
          <>
            <button className="trace-sql-toggle" onClick={() => setShowSql((s) => !s)}>
              {showSql ? "收起 SQL" : "查看 SQL"}
            </button>
            {showSql && <pre className="trace-sql">{item.sql}</pre>}
          </>
        )}
      </div>
    </div>
  );
}

/** 连续的主Agent取数工具（查本体口径/表结构/run_sql）归为一个可折叠执行块 */
function ToolGroup({ tools }: { tools: ToolNode[] }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="tool-group">
      <button className="tool-group-head" onClick={() => setOpen((o) => !o)}>
        <span className="tool-group-caret">{open ? "▾" : "▸"}</span>
        <span className="tool-group-title">取数执行 · 查阅本体口径并查询（{tools.length} 步）</span>
        <span className="tool-group-toggle">{open ? "收起" : "展开"}</span>
      </button>
      {open && (
        <div className="tool-group-body">
          {tools.map((t, i) => (
            <ToolBar key={i} item={t} compact />
          ))}
        </div>
      )}
    </div>
  );
}

type Block =
  | { t: "think"; node: Extract<TimelineNode, { kind: "think" }> }
  | { t: "answer"; node: Extract<TimelineNode, { kind: "answer" }> }
  | { t: "tool"; node: ToolNode }
  | { t: "group"; tools: ToolNode[] };

/** ReAct 内联时间线：主 Agent 的思考段、工具调用与直答按事件到达顺序自然穿插。 */
export function TraceStream({ items }: { items: TimelineNode[] }) {
  if (items.length === 0) return null;

  // 把相邻的主Agent取数工具聚成一组（中间夹思考/答案则断开）
  const blocks: Block[] = [];
  let buf: ToolNode[] = [];
  const flush = () => {
    if (buf.length === 1) blocks.push({ t: "tool", node: buf[0] });
    else if (buf.length > 1) blocks.push({ t: "group", tools: buf });
    buf = [];
  };
  for (const it of items) {
    if (it.kind === "tool" && it.skill_id === "query") {
      buf.push(it);
    } else {
      flush();
      if (it.kind === "think") blocks.push({ t: "think", node: it });
      else if (it.kind === "answer") blocks.push({ t: "answer", node: it });
      else blocks.push({ t: "tool", node: it }); // planner 的 list_skills 等单独展示
    }
  }
  flush();

  return (
    <div className="react-stream">
      {blocks.map((b, i) => {
        if (b.t === "think") {
          return (
            <p key={i} className="react-think">
              <span className="react-think-tag">思考</span>
              {b.node.text}
            </p>
          );
        }
        if (b.t === "answer") {
          return (
            <div key={i} className="main-answer">
              <Md text={b.node.text} />
            </div>
          );
        }
        if (b.t === "group") return <ToolGroup key={i} tools={b.tools} />;
        return <ToolBar key={i} item={b.node} />;
      })}
    </div>
  );
}
