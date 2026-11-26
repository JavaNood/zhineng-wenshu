import { useState } from "react";
import type { AgentCard, StepLog, TraceItem } from "../hooks/useQa";

const STATUS_LABEL: Record<string, string> = {
  waiting: "等待派发", running: "分析中", done: "完成", error: "错误", truncated: "超限中止",
};

function SqlFold({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  return (
    <div>
      <button type="button" className="trace-sql-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "收起 SQL" : "查看 SQL"}
      </button>
      {open && <pre className="step-sql">{text}</pre>}
    </div>
  );
}

function Thought({ text }: { text?: string }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  return (
    <div className="step-thought">
      <button type="button" className="trace-sql-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "收起思考" : "展开思考"}
      </button>
      {open && <p className="step-text">{text}</p>}
    </div>
  );
}

function ExpertBlock({ agent, steps }: { agent: AgentCard; steps: StepLog[] }) {
  const sqls = steps.filter((s) => s.kind === "sql");
  const plan = agent.plan || [];
  const stepDone = (n: number, action: string) =>
    sqls.some((s) => s.step === n || (s.action && s.action === action));
  const doneCount = plan.filter((p) => stepDone(p.n, p.action)).length;
  return (
    <div className="expert-block">
      <div className="expert-block-head">
        <span className="expert-name">{agent.skill_name}</span>
        <span className={`badge badge-${agent.status}`}>{STATUS_LABEL[agent.status]}</span>
        <span className="step-count">{plan.length ? `${doneCount}/${plan.length} 步` : `${sqls.length} 步`}</span>
      </div>

      <Thought text={agent.planThought} />

      {plan.length > 0 ? (
        <ol className="step-list">
          {plan.map((p) => {
            const done = sqls.find((s) => s.step === p.n || (s.action && s.action === p.action));
            return (
              <li key={p.n} className="step-row">
                <div className="step-head plan-head">
                  <span className="step-no">{p.n}</span>
                  <span className="step-title">{p.action}</span>
                  <span className="plan-flag">{done ? "已执行" : "待执行"}</span>
                </div>
                {done && (
                  <div className="step-detail">
                    <Thought text={done.thought} />
                    <SqlFold text={done.text} />
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="steps-empty">{agent.status === "running" ? "正在制定计划…" : "还没有计划"}</div>
      )}
    </div>
  );
}

export function ExpertSteps({
  agents, logs,
}: {
  agents: Record<string, AgentCard>;
  logs: Record<string, StepLog[]>;
  trace?: TraceItem[];
}) {
  const list = Object.values(agents).filter((a) => a.skill_id !== "query");
  if (list.length === 0) return null;
  return (
    <div className="expert-steps">
      {list.map((a) => (
        <ExpertBlock key={a.skill_id} agent={a} steps={logs[a.skill_id] || []} />
      ))}
    </div>
  );
}
