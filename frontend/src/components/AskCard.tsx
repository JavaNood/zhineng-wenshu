import { useEffect, useState } from "react";

export interface AskOption {
  key: string;
  kind: "expert" | "text" | "choice";
  label: string;
  skill_id?: string;
}

export interface AskCardData {
  card: "fuse" | "rename";
  prompt: string;
  options: AskOption[];
  skill_id?: string;
}

export function AskCard(props: {
  ask: AskCardData;
  error: string;
  onConfirm: (selected: string[], text: string) => void;
  onCancel: () => void;
  onOpenSkill?: (skillId: string) => void;
}) {
  const { ask, error, onConfirm, onCancel, onOpenSkill } = props;
  const [selected, setSelected] = useState<string[]>([]);
  const [text, setText] = useState("");
  const multi = ask.card === "fuse";

  useEffect(() => {
    setSelected([]);
    setText("");
  }, [ask]);

  const valueOf = (option: AskOption) => (option.kind === "expert" ? option.skill_id || option.key : option.key);

  const toggle = (option: AskOption) => {
    const value = valueOf(option);
    setText("");
    setSelected((cur) => {
      if (!multi) return [value];
      return cur.includes(value) ? cur.filter((item) => item !== value) : [...cur, value];
    });
  };

  const onText = (value: string) => {
    setText(value);
    if (!value.trim()) {
      setSelected((cur) => cur.filter((item) => item !== "custom"));
      return;
    }
    if (multi) {
      setSelected([]);
      return;
    }
    const textOption = ask.options.find((option) => option.kind === "text");
    setSelected(textOption ? [textOption.key] : []);
  };

  return (
    <section className="ask-card">
      {ask.card === "rename" && ask.skill_id && onOpenSkill ? (
        <button type="button" className="ask-prompt ask-prompt-link" onClick={() => onOpenSkill(ask.skill_id || "")}>
          {ask.prompt}
        </button>
      ) : (
        <p className="ask-prompt">{ask.prompt}</p>
      )}
      <div className="ask-options">
        {ask.options.map((option) => {
          const value = valueOf(option);
          const checked = option.kind === "text"
            ? text.trim().length > 0 && (multi || selected.includes(value))
            : selected.includes(value);
          return (
            <label className="ask-option" key={option.key}>
              <input
                type={multi ? "checkbox" : "radio"}
                name={multi ? undefined : "ask-option"}
                checked={checked}
                onChange={() => {
                  if (option.kind === "text") return;
                  toggle(option);
                }}
              />
              <span className="ask-key">{option.key}</span>
              <span className="ask-label">{option.label}</span>
              {option.kind === "text" && (
                <input
                  className="ask-text"
                  value={text}
                  placeholder="自己补充"
                  onChange={(e) => onText(e.target.value)}
                />
              )}
            </label>
          );
        })}
      </div>
      {error && <p className="ask-error">{error}</p>}
      <div className="ask-actions">
        <button type="button" className="ask-cancel" onClick={onCancel}>取消</button>
        <button type="button" className="ask-ok" onClick={() => onConfirm(selected, text.trim())}>确认</button>
      </div>
    </section>
  );
}
