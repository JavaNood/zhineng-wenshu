import { useState } from "react";
import type { AgentCard, StepLog, TraceItem } from "../hooks/useQa";

const STATUS_META: Record<string, { label: string; cls: string }> = {
  waiting: { label: "等待派发", cls: "st-waiting" },
  running: { label: "分析中", cls: "st-running" },
  done: { label: "已完成", cls: "st-done" },
  error: { label: "异常", cls: "st-error" },
  truncated: { label: "超限中止", cls: "st-error" },
};

function AgentCell({
  agent, logs, trace,
}: {
  agent: AgentCard;
  logs: Record<string, StepLog[]>;
  trace: TraceItem[];
}) {
  const [showPrompt, setShowPrompt] = useState(false);
  const [showThink, setShowThink] = useState(false);
  const steps = logs[agent.skill_id] || [];
  const sqlSteps = steps.filter((s) => s.kind === "sql");
  const plan = agent.plan || [];
  const stepDone = (n: number, action: string) =>
    sqlSteps.some((s) => s.step === n || (s.action && s.action === action));
  const doneCount = plan.filter((p) => stepDone(p.n, p.action)).length;
  const meta = STATUS_META[agent.status] || STATUS_META.waiting;
  // 该专家自己的取数工具调用（主链之外）
  const myRuns = trace.filter((t) => t.kind === "tool" && t.skill_id === agent.skill_id && t.tool === "run_sql");

  return (
    <div className={`agent-cell cell-${agent.status}`}>
      <div className="agent-cell-head">
        <span className="cell-dot" />
        <span className="cell-name">{agent.skill_name}</span>
        <span className={`cell-status ${meta.cls}`}>{meta.label}</span>
        {agent.status === "running" && agent.step > 0 && (
          <span className="cell-step">第 {agent.step} 步</span>
        )}
        <span className="cell-prompt-toggle" onClick={() => setShowPrompt((s) => !s)}>
          {showPrompt ? "收起任务书" : "任务书"}
        </span>
      </div>

      {showPrompt && agent.prompt && (
        <pre className="cell-prompt">{agent.prompt}</pre>
      )}

      <div className="agent-cell-body">
        {agent.status === "running" && (
          <div className="cell-live">
            <span className="spinner" />
            {myRuns.length > 0 && (
              <div className="cell-runs">已执行 {myRuns.length} 次查询</div>
            )}
          </div>
        )}

        {(agent.plan?.length || sqlSteps.length > 0) && (
          <div className="cell-thinkbox">
            <button type="button" className="cell-think-toggle" onClick={() => setShowThink((s) => !s)}>
              <span className="cell-think-arrow">{showThink ? "▼" : "▶"}</span>
              计划 {plan.length ? doneCount : sqlSteps.length}/{plan.length || sqlSteps.length}
            </button>
            {showThink && (
              <div className="cell-thinkbody">
                {(plan.length ? plan : sqlSteps.map((s, i) => ({ n: s.step || i + 1, action: s.action || "" }))).map((p) => (
                  <div key={p.n + p.action} className="cell-steprow kind-sql">
                    <span className="cell-stepmark">{stepDone(p.n, p.action) ? "⚙️" : "○"}</span>
                    <span className="cell-steptext">{p.action}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** 被主 Agent 派出的分析专家：每栏独立状态与自身思路，无生硬大标题，自然续接主思维链。 */
export function SubagentBoard({
  agents, logs, trace,
}: {
  agents: Record<string, AgentCard>;
  logs: Record<string, StepLog[]>;
  trace: TraceItem[];
}) {
  const list = Object.values(agents).filter((a) => a.skill_id !== "query");
  if (list.length === 0) return null;
  return (
    <section className="block agent-board">
      <div className="agent-grid">
        {list.map((a) => (
          <AgentCell key={a.skill_id} agent={a} logs={logs} trace={trace} />
        ))}
      </div>
    </section>
  );
}
