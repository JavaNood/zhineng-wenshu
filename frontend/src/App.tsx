import { useEffect, useMemo, useState } from "react";
import { useQa } from "./hooks/useQa";
import { ModelPicker } from "./components/ModelPicker";
import { TraceStream } from "./components/TraceStream";
import { SubagentBoard } from "./components/SubagentBoard";
import { ExpertSteps } from "./components/ExpertSteps";
import { EvidenceAnchor } from "./components/EvidenceAnchor";
import { Md } from "./components/Md";

/** 兜底：旧措辞「该 Skill 视角下的主要原因」统一为「该专家的主要结论」。 */
const fixTitle = (s?: string) =>
  (s || "").replace(/该\s*Skill\s*视角下的主要原因/g, "该专家的主要结论");

export default function App() {
  const q = useQa();
  const [question, setQuestion] = useState("");
  const [sideOpen, setSideOpen] = useState(true);
  const [round, setRound] = useState<{ q: string } | null>(null);

  useEffect(() => { q.loadModels(); }, []);

  const submit = () => {
    if (!question.trim() || q.busy) return;
    setRound({ q: question.trim() });
    q.ask(question.trim());
    setQuestion("");
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  // 真实分析专家（问数由主 Agent 直答，不进此列）：结论区、右侧步骤、共识分歧只认这批
  const expertAgents = Object.values(q.agents).filter(
    (a) => a.skill_id !== "query" && a.status === "done" && a.conclusion
  );
  const hasExperts = expertAgents.length > 0;
  const hasAnyExpert = Object.values(q.agents).some((a) => a.skill_id !== "query");
  // 主对话时间线按派工点切分：派工前（问数/表态/选专家）在工作区前，派工后（完成播报）在工作区后
  const beforeNodes = useMemo(() => q.timeline.slice(0, q.dispatchAt), [q.timeline, q.dispatchAt]);
  const afterNodes = useMemo(() => q.timeline.slice(q.dispatchAt), [q.timeline, q.dispatchAt]);

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
            {!round && !q.busy && (
              <div className="welcome">
                <h2>我是智能问数助手</h2>
                <p>我能帮你查询23-25年中国新零售经营分析数据结果并进行分析，你可以试着问我。</p>
                <div className="examples">
                  <button onClick={() => setQuestion("2025年的销售额是多少？")}>2025年的销售额是多少？</button>
                  <button onClick={() => setQuestion("2025年4季度、3季度销售额分别是多少，变化的原因是什么？")}>2025Q4 销售额为什么下降了？</button>
                </div>
              </div>
            )}

            {round && (
              <div className="msg msg-user">
                <div className="bubble bubble-user">{round.q}</div>
              </div>
            )}

            {q.error && <div className="error-box">⚠ {q.error}</div>}
            {!q.keyOk && <div className="warn-box">未检测到 DEEPSEEK_API_KEY，请在后端 .env 配置后重启。</div>}

            {/* 助手回答区 */}
            {(q.busy || q.summary || hasExperts || q.timeline.length > 0 || q.chatAnswer) && (
              <div className="msg msg-bot">
                <div className="bubble bubble-bot">
                  {/* 派工前：问数取数、主Agent表态、通读本领、挑选专家 */}
                  <TraceStream items={beforeNodes} />
                  {q.chatAnswer && <p className="chat-answer">{q.chatAnswer}</p>}

                  {/* 被派出的分析专家：各栏独立状态与自身思路，自然续接主思维链 */}
                  <SubagentBoard agents={q.agents} logs={q.logs} trace={q.trace} />

                  {/* 派工后：专家全部完成，主Agent宣布汇总，再呈现结果 */}
                  <TraceStream items={afterNodes} />

                  {/* 各专家结论：逐专家纵向组织输出（不再三列并排），Markdown 渲染 */}
                  {hasExperts && (
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

                  {/* 共识 / 分歧，仅分析专家 */}
                  {hasExperts && (q.consensus.length > 0 || q.divergences.length > 0) && (
                    <section className="block">
                      <h3>共识与分歧</h3>
                      {q.consensus.length > 0 && (
                        <div className="consensus">
                          <b>共识（多方法共同指向）：</b>
                          <ul>{q.consensus.map((c, i) => (
                            <li key={i}><Md text={c.point} /> <em>（{c.support?.join("、")}）</em></li>
                          ))}</ul>
                        </div>
                      )}
                      {q.divergences.length > 0 && (
                        <div className="divergence">
                          <b>分歧（待复核，不强行调和）：</b>
                          <ul>{q.divergences.map((d, i) => (
                            <li key={i}><Md text={d.point} /> — {d.why}</li>
                          ))}</ul>
                        </div>
                      )}
                    </section>
                  )}

                  {/* 汇总答案，仅分析专家 */}
                  {hasExperts && q.summary && (
                    <section className="block summary-card">
                      <h3>汇总结论</h3>
                      <Md className="summary-text" text={fixTitle(q.summary)} />
                    </section>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* 底部输入框 */}
          <div className="composer">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
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
