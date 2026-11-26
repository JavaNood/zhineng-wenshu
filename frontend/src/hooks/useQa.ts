import { useCallback, useEffect, useRef, useState } from "react";
import { fetchModels, startChat, connectStream, postReply } from "../api";

export interface AgentCard {
  skill_id: string;
  skill_name: string;
  status: "waiting" | "running" | "done" | "error" | "truncated";
  step: number;
  conclusion?: string;
  evidences?: any[];
  prompt?: string;
  plan?: { n: number; action: string }[];
  planThought?: string;
}

export interface StepLog {
  step: number;
  kind: "sql" | "think" | "conclusion";
  text: string;
  action?: string;
  /** 这一步取数时模型的思维链，挂在计划步骤下面，不当成新的步骤。 */
  thought?: string;
}

/** 统一过程时间线：思考(think) 与 工具调用(tool) 按到达顺序交错记录。 */
export type TraceItem =
  | { kind: "think"; skill_id: string; text: string }
  | { kind: "tool"; skill_id: string; tool: string; title: string; detail?: string; sql?: string; body?: string };

/** 主对话时间线节点：思考 / 工具调用 / 主Agent直答，按事件到达顺序追加。 */
export type TimelineNode =
  | { kind: "think"; skill_id: string; text: string }
  | { kind: "act"; skill_id: string; text: string }
  | { kind: "tool"; skill_id: string; tool: string; title: string; detail?: string; sql?: string; body?: string }
  | { kind: "answer"; text: string };

export interface EvidenceView {
  metric: string;
  formula?: string;
  sql: string;
  row_count: number;
  rows: any[];
}

export function useQa() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("deepseek-reasoner");
  const [keyOk, setKeyOk] = useState(true);
  const [busy, setBusy] = useState(false);
  const [intent, setIntent] = useState<any[]>([]);
  const [plan, setPlan] = useState<any>(null);
  const [agents, setAgents] = useState<Record<string, AgentCard>>({});
  const [logs, setLogs] = useState<Record<string, StepLog[]>>({});
  const [trace, setTrace] = useState<TraceItem[]>([]);
  // 主对话时间线节点流 + 派工切分点（首个 agent_start 到达时的节点长度）
  const [timeline, setTimeline] = useState<TimelineNode[]>([]);
  const [dispatchAt, setDispatchAt] = useState(0);
  const tlRef = useRef<TimelineNode[]>([]);
  const dispatchedRef = useRef(false);
  const [chatAnswer, setChatAnswer] = useState("");
  const [mainAnswer, setMainAnswer] = useState("");
  const [consensus, setConsensus] = useState<any[]>([]);
  const [divergences, setDivergences] = useState<any[]>([]);
  const [summary, setSummary] = useState("");
  const [evidences, setEvidences] = useState<EvidenceView[]>([]);
  const [tokens, setTokens] = useState<any>(null);
  const [error, setError] = useState("");
  const [ask, setAsk] = useState<any>(null);
  const [askError, setAskError] = useState("");
  const [askStatus, setAskStatus] = useState("");
  const [created, setCreated] = useState<{ skillId: string; title: string } | null>(null);
  const [skillView, setSkillView] = useState<{ title: string; text: string } | null>(null);
  const closeRef = useRef<() => void>(() => {});
  const sessionRef = useRef("");
  const askRef = useRef<any>(null);

  const loadModels = useCallback(async () => {
    const m = await fetchModels();
    setModels(m.models);
    setModel(m.default);
    setKeyOk(m.key_configured);
  }, []);

  const handleEvent = useCallback((ev: any) => {
    switch (ev.type) {
      case "intent":
        setIntent(ev.points);
        break;
      case "plan":
        setPlan(ev.plan);
        break;
      case "agent_start":
        if (ev.skill_id === "query") break; // 问数由主 Agent 直答，不建专家卡
        if (!dispatchedRef.current) {
          dispatchedRef.current = true;
          setDispatchAt(tlRef.current.length);
        }
        setAgents((a) => ({
          ...a,
          [ev.skill_id]: { skill_id: ev.skill_id, skill_name: ev.skill_name, status: "waiting", step: 0 },
        }));
        break;
      case "agent_progress":
        if (ev.skill_id === "query") break;
        setAgents((a) => ({
          ...a,
          [ev.skill_id]: { ...a[ev.skill_id], step: ev.step, status: "running" },
        }));
        break;
      case "step_log":
        if (ev.skill_id === "query") break;
        setLogs((l) => ({
          ...l,
          [ev.skill_id]: [...(l[ev.skill_id] || []), {
            step: ev.step, kind: ev.kind, text: ev.text, action: ev.action, thought: ev.thought,
          }],
        }));
        break;
      case "worker_plan":
        setAgents((a) => ({
          ...a,
          [ev.skill_id]: {
            ...a[ev.skill_id],
            plan: ev.steps || [],
            planThought: ev.thought || "",
          },
        }));
        break;
      case "act": {
        // 和思考一样按增量拼接，避免每个 token 变成一块。到达即展示。
        const piece = ev.text || "";
        const sid = ev.skill_id || "";
        if (sid === "main" || sid === "planner") {
          const cur = tlRef.current;
          const last = cur[cur.length - 1];
          tlRef.current = last && last.kind === "act" && last.skill_id === sid
            ? [...cur.slice(0, -1), { ...last, text: last.text + piece }]
            : [...cur, { kind: "act", skill_id: sid, text: piece }];
          setTimeline(tlRef.current);
        }
        break;
      }
      case "reasoning": {
        // 主 Agent 一轮一个思考段。专家的思维链挂在计划步骤上，不进主时间线，也不拆成步骤。
        const piece = ev.text || "";
        const sid = ev.skill_id || "";
        if (sid === "main" || sid === "planner" || sid === "query") {
          const cur = tlRef.current;
          const last = cur[cur.length - 1];
          tlRef.current = last && last.kind === "think" && last.skill_id === sid
            ? [...cur.slice(0, -1), { ...last, text: last.text + piece }]
            : [...cur, { kind: "think", skill_id: sid, text: piece }];
          setTimeline(tlRef.current);
        }
        break;
      }
      case "tool_call":
        // 主 Agent 的工具进主时间线；专家工具归各自栏
        if (ev.skill_id === "main" || ev.skill_id === "planner" || ev.skill_id === "query") {
          tlRef.current = [...tlRef.current, {
            kind: "tool", skill_id: ev.skill_id || "", tool: ev.tool, title: ev.title,
            detail: ev.detail, sql: ev.sql, body: ev.body,
          }];
          setTimeline(tlRef.current);
        }
        setTrace((t) => [...t, {
          kind: "tool", skill_id: ev.skill_id || "", tool: ev.tool, title: ev.title,
          detail: ev.detail, sql: ev.sql, body: ev.body,
        }]);
        break;
      case "agent_prompt":
        if (ev.skill_id === "query") break;
        setAgents((a) => ({
          ...a,
          [ev.skill_id]: { ...a[ev.skill_id], prompt: ev.prompt },
        }));
        break;
      case "sub_done": {
        const s = ev.sub;
        setAgents((a) => ({
          ...a,
          [s.skill_id]: {
            skill_id: s.skill_id, skill_name: s.skill_name,
            status: s.status === "done" ? "done" : s.status,
            step: a[s.skill_id]?.step ?? 0,
            conclusion: s.conclusion, evidences: s.evidences,
            prompt: a[s.skill_id]?.prompt,
            plan: a[s.skill_id]?.plan,
            planThought: a[s.skill_id]?.planThought,
          },
        }));
        break;
      }
      case "chat_answer":
        setChatAnswer(ev.text || "");
        setBusy(false);
        break;
      case "query_answer":
        // 问数由主 Agent 直答：作为主对话时间线节点插入，不进专家体系
        tlRef.current = [...tlRef.current, { kind: "answer", text: ev.text || "" }];
        setTimeline(tlRef.current);
        setEvidences((prev) => [...prev, ...(ev.evidences || [])]);
        break;
      case "consensus":
        setConsensus(ev.consensus || []);
        setDivergences(ev.divergences || []);
        break;
      case "final":
        setSummary(ev.summary);
        setEvidences(ev.evidences || []);
        break;
      case "tokens":
        setTokens(ev.tokens);
        break;
      case "ask":
        askRef.current = ev;
        setAsk(ev);
        setAskError("");
        setAskStatus("");
        setBusy(true);
        break;
      case "ask_status":
        askRef.current = null;
        setAsk(null);
        setAskStatus(ev.text || "");
        setBusy(true);
        break;
      case "ask_closed":
        askRef.current = null;
        setAsk(null);
        setAskStatus("");
        if (ev.card === "rename" && ev.skill_id) {
          setCreated({
            skillId: ev.skill_id,
            title: ev.title || "新专家",
          });
        }
        break;
      case "error":
        setError(ev.message);
        askRef.current = null;
        setAsk(null);
        setAskStatus("");
        setBusy(false);
        closeRef.current?.();
        break;
    }
  }, []);

  const askQuestion = useCallback(async (question: string, ontologyEnabled = true) => {
    setBusy(true);
    setError(""); setChatAnswer(""); setMainAnswer("");
    setConsensus([]); setDivergences([]); setSummary(""); setEvidences([]);
    setTokens(null); setAgents({}); setLogs({}); setIntent([]); setPlan(null);
    setAsk(null); setAskError(""); setAskStatus(""); setCreated(null);
    askRef.current = null;
    setTrace([]);
    tlRef.current = [];
    dispatchedRef.current = false;
    setTimeline([]);
    setDispatchAt(0);
    try {
      let start;
      try {
        start = await startChat(question, model, ontologyEnabled, sessionRef.current || undefined);
      } catch (err) {
        if (!sessionRef.current) throw err;
        sessionRef.current = "";
        start = await startChat(question, model, ontologyEnabled);
      }
      sessionRef.current = start.session_id;
      closeRef.current = connectStream(start.session_id, handleEvent, () => {
        if (!askRef.current) setBusy(false);
      });
    } catch (e: any) {
      setError(String(e));
      setBusy(false);
    }
  }, [model, handleEvent]);

  const reply = useCallback(async (selected: string[], text: string, action = "confirm") => {
    const id = sessionRef.current;
    const card = askRef.current?.card;
    if (!id || !card) return;
    setAskError("");
    setBusy(true);
    try {
      const res = await postReply(id, { card, action, selected, text });
      if (!res.accepted) {
        setAskError(res.error || "没有通过");
        return;
      }
      if (res.done) {
        askRef.current = null;
        setAsk(null);
        setAskStatus("");
        setBusy(false);
        return;
      }
      askRef.current = null;
      setAsk(null);
      setAskStatus(card === "fuse" ? "正在调用 skills 构建新的专家，请等待..." : "");
      closeRef.current = connectStream(id, handleEvent, () => {
        if (!askRef.current) setBusy(false);
      });
    } catch (e: any) {
      setAskError(String(e.message || e));
    }
  }, [handleEvent]);

  useEffect(() => {
    const onHide = () => {
      const id = sessionRef.current;
      if (!id) return;
      navigator.sendBeacon(`/api/session/${id}/end`);
    };
    window.addEventListener("pagehide", onHide);
    return () => window.removeEventListener("pagehide", onHide);
  }, []);

  const openSkill = useCallback(async (skillId: string) => {
    const id = sessionRef.current;
    if (!id || !skillId) return;
    const r = await fetch(`/api/session/${id}/skill/${skillId}`);
    const data = await r.json();
    if (!r.ok) return;
    setSkillView({ title: data.title || skillId, text: data.text || "" });
  }, []);

  const closeSkill = useCallback(() => setSkillView(null), []);

  const reset = useCallback(() => {
    closeRef.current?.();
    setBusy(false);
  }, []);

  return {
    models, model, setModel, keyOk, busy, intent, plan, agents, logs, trace,
    timeline, dispatchAt,
    chatAnswer, mainAnswer, consensus, divergences, summary, evidences, tokens, error,
    askCard: ask, askError, askStatus, created, skillView,
    loadModels, ask: askQuestion, reply, openSkill, closeSkill, reset,
  };
}
