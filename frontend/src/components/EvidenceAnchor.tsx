import { useState } from "react";
import type { EvidenceView } from "../hooks/useQa";

function EvidenceItem({ ev, idx }: { ev: EvidenceView; idx: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="evidence-item">
      <button className="evidence-anchor" onClick={() => setOpen((o) => !o)}>
        [{idx + 1}] {ev.metric || "取数证据"}（{ev.row_count} 行）
      </button>
      {open && (
        <div className="evidence-body">
          {ev.formula && <div className="ev-formula">口径：{ev.formula}</div>}
          <pre className="ev-sql">{ev.sql}</pre>
          {ev.rows?.length > 0 && (
            <div className="ev-table-wrap">
              <table className="ev-table">
                <thead>
                  <tr>{Object.keys(ev.rows[0]).map((k) => <th key={k}>{k}</th>)}</tr>
                </thead>
                <tbody>
                  {ev.rows.slice(0, 10).map((r, i) => (
                    <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function EvidenceAnchor({ evidences }: { evidences: EvidenceView[] }) {
  if (!evidences?.length) return null;
  return (
    <div className="evidence-block">
      <h3>可验证证据（点击展开 SQL / 结果）</h3>
      {evidences.map((ev, i) => <EvidenceItem key={i} ev={ev} idx={i} />)}
    </div>
  );
}
