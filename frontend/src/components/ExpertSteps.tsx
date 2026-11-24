import { useState } from "react";
import type { AgentCard, StepLog, TraceItem } from "../hooks/useQa";

const STATUS_LABEL: Record<string, string> = {
  waiting: "等待派发", running: "分析中", done: "完成", error: "错误", truncated: "超限中止",
};

const KIND_FALLBACK: Record<string, string> = {
  sql: "执行数据查询", think: "分析判断", conclusion: "形成结论",
};

/** 清洗模型给出的动作句：丢弃残句（未闭合括号、过短碎片），避免右侧出现"订单量）"这类半截标题。 */
function cleanAction(action?: string): string | null {
  if (!action) return null;
  const s = action.trim();
  if (s.length < 4) return null;
  // 括号不配对视为残句
  if ((s.match(/（/g) || []).length !== (s.match(/）/g) || []).length) return null;
  if ((s.match(/\(/g) || []).length !== (s.match(/\)/g) || []).length) return null;
  // 去掉残留的前缀序号
  return s.replace(/^步骤\s*[0-9一二三四五六七八九十]+[：:.\s]*/, "").trim() || null;
}

function StepRow({ log, index }: { log: StepLog; index: number }) {
  const [open, setOpen] = useState(false);
  const clean = cleanAction(log.action);
  const title = clean || `${index + 1}. ${KIND_FALLBACK[log.kind] || "执行一步"}`;
  const isSql = log.kind === "sql";
  return (
    <li className="step-row">
      <button className="step-head" onClick={() => setOpen((o) => !o)}>
        <span className="step-no">{index + 1}</span>
        <span className="step-title">{title}</span>
        <span className="step-caret">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="step-detail">
          {isSql ? (
            <pre className="step-sql">{log.text}</pre>
          ) : (
            <p className="step-text">{log.text}</p>
          )}
        </div>
      )}
    </li>
  );
}

function ExpertBlock({ agent, steps, tools }: {
  agent: AgentCard; steps: StepLog[]; tools: TraceItem[];
}) {
  // 右侧只展示取数/判断动作步骤，结论已在主结论区渲染
  const actionSteps = steps.filter((s) => s.kind !== "conclusion");
  const ontologyCalls = tools.filter(
    (t): t is Extract<TraceItem, { kind: "tool" }> => t.kind === "tool" && t.tool !== "run_sql"
  );
  return (
    <div className="expert-block">
      <div className="expert-block-head">
        <span className="expert-name">{agent.skill_name}</span>
        <span className={`badge badge-${agent.status}`}>{STATUS_LABEL[agent.status]}</span>
        <span className="step-count">{actionSteps.length} 步</span>
      </div>

      {ontologyCalls.length > 0 && (
        <div className="expert-ontos">
          {ontologyCalls.map((t, i) => (
            <div key={i} className="onto-line">
              <span className="onto-icon">📚</span>
              <span>{t.title}：{t.detail}</span>
            </div>
          ))}
        </div>
      )}

      {actionSteps.length > 0 ? (
        <ol className="step-list">
          {actionSteps.map((s, i) => <StepRow key={i} log={s} index={i} />)}
        </ol>
      ) : (
        <div className="steps-empty">等待执行步骤…</div>
      )}
    </div>
  );
}

export function ExpertSteps({
  agents, logs, trace,
}: {
  agents: Record<string, AgentCard>;
  logs: Record<string, StepLog[]>;
  trace: TraceItem[];
}) {
  const list = Object.values(agents).filter((a) => a.skill_id !== "query");
  if (list.length === 0) return null;
  return (
    <div className="expert-steps">
      {list.map((a) => (
        <ExpertBlock
          key={a.skill_id}
          agent={a}
          steps={logs[a.skill_id] || []}
          tools={trace.filter((t) => t.skill_id === a.skill_id)}
        />
      ))}
    </div>
  );
}
