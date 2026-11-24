export function ModelPicker({
  models, model, onChange, disabled,
}: {
  models: string[]; model: string; onChange: (m: string) => void; disabled?: boolean;
}) {
  return (
    <label className="model-picker">
      模型
      <select value={model} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {models.map((m) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
    </label>
  );
}
