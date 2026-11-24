"""从模型回复里抽取 SQL / 结论 / JSON，以及步骤标题。行为与迁移前一致。"""
from __future__ import annotations
import json
import re


def _extract_json(text: str) -> dict | list:
    """从 LLM 输出中提取 JSON（容忍 ```json 包裹与前后说明）。"""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    cand = m.group(1) if m else text
    cand = cand.strip()
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        # 截取首个 { 到最后一个 }
        a, b = cand.find("{"), cand.rfind("}")
        if a != -1 and b != -1:
            try:
                return json.loads(cand[a:b + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _extract_block(text: str, tag: str) -> str | None:
    m = re.search(rf"```{tag}\s*([\s\S]*?)```", text or "")
    return m.group(1).strip() if m else None


def _extract_sql(text: str) -> str | None:
    m = re.search(r"```sql\s*([\s\S]*?)```", text or "")
    return m.group(1).strip() if m else None


def _model_line(text: str) -> str | None:
    """取模型自己写出的第一句，用作步骤标题。"""
    for ln in (text or "").splitlines():
        s = ln.strip().lstrip("#").strip()
        if len(s) >= 4 and not s.startswith("```"):
            return s[:80]
    return None


def _action_before_sql(text: str, sql: str | None = None) -> str | None:
    """提取 SQL 代码块之前的一句中文动作说明，作为右侧步骤标题。"""
    if not text:
        return None
    pre = text.split("```sql")[0] if "```sql" in text else text
    lines = [ln.strip(" ：:；;·-") for ln in pre.splitlines() if ln.strip()]
    for ln in reversed(lines):
        cn = re.findall(r"[一-鿿][一-鿿，、：:？?！!（）()0-9A-Za-z.%\-]+", ln)
        if cn:
            s = cn[-1].strip()
            if len(s) >= 4:
                return s[:60]
    return None


def _rows_preview(rows: list[dict]) -> str:
    if not rows:
        return "(空结果)"
    cols = list(rows[0].keys())
    out = [" | ".join(cols)]
    for r in rows:
        out.append(" | ".join(str(r.get(c, "")) for c in cols))
    return "\n".join(out)
