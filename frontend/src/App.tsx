import { useEffect, useMemo, useRef, useState } from "react";
import { useQa, type AgentCard, type StepLog, type TimelineNode, type TraceItem } from "./hooks/useQa";
import { ModelPicker } from "./components/ModelPicker";
import { TraceStream } from "./components/TraceStream";
import { SubagentBoard } from "./components/SubagentBoard";
import { ExpertSteps } from "./components/ExpertSteps";
import { EvidenceAnchor } from "./components/EvidenceAnchor";
import { Md } from "./components/Md";

/** 兜底：旧措辞「该 Skill 视角下的主要原因」统一为「该专家的主要结论」。 */
const fixTitle = (s?: string) =>
  (s || "").replace(/该\s*Skill\s*视角下的主要原因/g, "该专家的主要结论");

/** 一轮已经结束的对话。再提问时整段留在上面，不跟当前这一轮混在一起。 */
interface PastTurn {
  question: string;
  error: string;
  timeline: TimelineNode[];
  dispatchAt: number;
  agents: Record<string, AgentCard>;
  logs: Record<string, StepLog[]>;
  trace: TraceItem[];
  chatAnswer: string;
  consensus: any[];
  divergences: any[];
  summary: string;
}

function TurnThread(props: PastTurn & { busy: boolean }) {
  const { question, error, busy, timeline, dispatchAt, agents, logs, trace,
    chatAnswer, consensus, divergences, summary } = props;
  const expertAgents = Object.values(agents).filter(
    (a) => a.skill_id !== "query" && a.status === "done" && a.conclusion
  );
  const hasExperts = expertAgents.length > 0;
  const beforeNodes = useMemo(
    () => (dispatchAt > 0 ? timeline.slice(0, dispatchAt) : timeline),
    [timeline, dispatchAt],
  );
  const afterNodes = useMemo(
    () => (dispatchAt > 0 ? timeline.slice(dispatchAt) : []),
    [timeline, dispatchAt],
  );
  const introAct = afterNodes.findIndex((n) => n.kind === "act");
  const conclusionsReady = introAct >= 0 && (introAct < afterNodes.length - 1 || !busy || Boolean(summary));
  const leadNodes = conclusionsReady ? afterNodes.slice(0, introAct + 1) : afterNodes;
  const tailNodes = conclusionsReady ? afterNodes.slice(introAct + 1) : [];
  const showBot = busy || summary || hasExperts || timeline.length > 0 || chatAnswer;
  return (
    <>
      <div className="msg msg-user">
        <div className="bubble bubble-user">{question}</div>
      </div>
      {error && <div className="error-box">⚠ {error}</div>}
      {showBot && (
        <div className="msg msg-bot">
          <div className="bubble bubble-bot">
            <TraceStream items={beforeNodes} />
            {chatAnswer && <p className="chat-answer">{chatAnswer}</p>}
            <SubagentBoard agents={agents} logs={logs} trace={trace} />
            <TraceStream items={leadNodes} />
            {conclusionsReady && hasExperts && (
              <section className="block">
                <h3>各专家诊断结论</h3>
                <div className="expert-list">
                  {expertAgents.map((a) => (
                    <div key={a.skill_id} className="expert-card">
                      <div className="expert-name">
                        <span className="expert-badge">专家</span>
                        {a.skill_name}
                      </div>
                      <Md className="expert-conclusion" text={fixTitle(a.conclusion)} />
                    </div>
                  ))}
                </div>
              </section>
            )}
            <TraceStream items={tailNodes} />
            {hasExperts && (consensus.length > 0 || divergences.length > 0) && (
              <section className="block">
                <h3>共识与分歧</h3>
                {consensus.length > 0 && (
                  <div className="consensus">
                    <b>共识（多方法共同指向）：</b>
                    <ul>{consensus.map((c, i) => (
                      <li key={i}><Md text={c.point} /> <em>（{c.support?.join("、")}）</em></li>
                    ))}</ul>
                  </div>
                )}
                {divergences.length > 0 && (
                  <div className="divergence">
                    <b>分歧（待复核，不强行调和）：</b>
                    <ul>{divergences.map((d, i) => (
                      <li key={i}><Md text={d.point} /> — {d.why}</li>
                    ))}</ul>
                  </div>
                )}
              </section>
            )}
            {hasExperts && summary && (
              <section className="block summary-card">
                <h3>汇总结论</h3>
                <Md className="summary-text" text={fixTitle(summary)} />
              </section>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default function App() {
  const q = useQa();
  const [question, setQuestion] = useState("");
  const [sideOpen, setSideOpen] = useState(true);
  const [round, setRound] = useState<{ q: string } | null>(null);
  const [history, setHistory] = useState<PastTurn[]>([]);
  // macOS 中文输入法：选词回车会先结束组词，紧接着再来一次 Enter。这一拍不能当成发送。
  const imeLock = useRef(false);

  useEffect(() => { q.loadModels(); }, []);

  const submit = () => {
    const text = question.trim();
    if (!text || q.busy) return;
    if (round) {
      setHistory((h) => [...h, {
        question: round.q,
        error: q.error,
        timeline: q.timeline,
        dispatchAt: q.dispatchAt,
        agents: q.agents,
        logs: q.logs,
        trace: q.trace,
        chatAnswer: q.chatAnswer,
        consensus: q.consensus,
        divergences: q.divergences,
        summary: q.summary,
      }]);
    }
    setRound({ q: text });
    q.ask(text);
    setQuestion("");
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== "Enter" || e.shiftKey) return;
    if (e.nativeEvent.isComposing || e.nativeEvent.keyCode === 229 || imeLock.current) return;
    e.preventDefault();
    submit();
  };

  const hasAnyExpert = Object.values(q.agents).some((a) => a.skill_id !== "query");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">智能问数</div>
        <div className="topbar-right">
          <ModelPicker models={q.models} model={q.model} onChange={q.setModel} disabled={q.busy} />
        </div>
      </header>

      <div className="chat-layout">
        {/* 主聊天流 */}
        <main className="chat-main">
          <div className="msg-list">
            {history.length === 0 && !round && !q.busy && (
              <div className="welcome">
                <h2>我是智能问数助手</h2>
                <p>我能帮你查询23-25年中国新零售经营分析数据结果并进行分析，你可以试着问我。</p>
                <div className="examples">
                  <button onClick={() => setQuestion("2025年的销售额是多少？")}>2025年的销售额是多少？</button>
                  <button onClick={() => setQuestion("2025年1季度和2季度的销售额是多少，变化的原因是什么？")}>2025年1季度和2季度的销售额是多少，变化的原因是什么？</button>
                </div>
              </div>
            )}

            {history.map((h, i) => (
              <TurnThread key={i} question={h.question} error={h.error} busy={false}
                timeline={h.timeline} dispatchAt={h.dispatchAt} agents={h.agents}
                logs={h.logs} trace={h.trace} chatAnswer={h.chatAnswer}
                consensus={h.consensus} divergences={h.divergences} summary={h.summary} />
            ))}

            {round && (
              <TurnThread question={round.q} error={q.error} busy={q.busy}
                timeline={q.timeline} dispatchAt={q.dispatchAt} agents={q.agents}
                logs={q.logs} trace={q.trace} chatAnswer={q.chatAnswer}
                consensus={q.consensus} divergences={q.divergences} summary={q.summary} />
            )}

            {!q.keyOk && <div className="warn-box">未检测到 DEEPSEEK_API_KEY，请在后端 .env 配置后重启。</div>}
          </div>

          {/* 底部输入框 */}
          <div className="composer">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onCompositionStart={() => { imeLock.current = true; }}
              onCompositionEnd={() => {
                imeLock.current = true;
                window.setTimeout(() => { imeLock.current = false; }, 0);
              }}
              onKeyDown={onKey}
              placeholder="输入经营问题，Enter 发送，Shift+Enter 换行…"
              rows={2}
              disabled={q.busy}
            />
            <button className="send-btn" onClick={submit} disabled={q.busy || !q.keyOk || !question.trim()}>
              {q.busy ? "分析中…" : "发送"}
            </button>
          </div>
        </main>

        {/* 右侧过程窗口（可展开/折叠）：仅放各专家执行步骤与证据，规划思考在主对话流 */}
        <aside className={`side ${sideOpen ? "open" : "closed"}`}>
          <button className="side-toggle" onClick={() => setSideOpen((o) => !o)}>
            {sideOpen ? "◀ 收起过程" : "过程 ▶"}
          </button>
          {sideOpen && (
            <div className="side-body">
              {hasAnyExpert && (
                <section className="side-sec">
                  <h4>各专家执行步骤</h4>
                  <ExpertSteps agents={q.agents} logs={q.logs} trace={q.trace} />
                </section>
              )}

              <section className="side-sec">
                <h4>可验证证据</h4>
                <EvidenceAnchor evidences={q.evidences} />
              </section>

              {q.tokens && (
                <footer className="tokens">
                  Tokens 输入 {q.tokens.prompt} · 输出 {q.tokens.completion} · 合计 {q.tokens.total}
                </footer>
              )}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
