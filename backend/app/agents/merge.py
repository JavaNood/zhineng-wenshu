"""汇聚节点使用的共识 / 分歧合并。提示词正文与迁移前一致。"""
from __future__ import annotations
import re
from .. import llm
from ..schemas import Evidence, FinalResult, SubResult
from .planner import _extract_json


# 提示词正文不改，只是从 supervisor 挪到汇聚节点旁边。
MERGE_SYS = """你是结果汇聚者。阅读各专家子 Agent 的结论，输出 JSON：
{"summary":"整体结论","consensus":[{"point":"多方法共同指向的判断","support":["来源skill"]}],"divergences":[{"point":"结论不一致处","why":"分歧原因","pending":"待复核"}]}}
规则：共识=≥2个方法共同指向；分歧=结论冲突，挂起待复核，不强行调和；不新增无证据的判断。只输出 JSON。"""


async def merge_results(
    sub_results: list[SubResult], model: str, on_reasoning=None,
) -> tuple[FinalResult, dict]:
    local_usage = {"prompt": 0, "completion": 0, "total": 0}
    blocks = "\n\n".join(
        f"【{r.skill_name}】({r.status})\n{r.conclusion}" for r in sub_results)
    resp = await llm.chat(
        [{"role": "system", "content": MERGE_SYS},
         {"role": "user", "content": blocks}],
        model=model, on_reasoning=on_reasoning)
    for k in ("prompt", "completion", "total"):
        local_usage[k] += resp.get("usage", {}).get(k, 0)
    raw = resp.get("content", "") or ""
    data = _extract_json(raw) or {}
    summary = (data.get("summary") or "").strip()
    consensus = data.get("consensus") or []
    divergences = data.get("divergences") or []
    # 宽松兜底：标准 JSON 未完整解析时，从原文正则补提 summary 字段
    if not summary:
        m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.S)
        if m:
            summary = m.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
    # 解析不到 summary 字段时，展示模型原文，不另写一句汇总。
    if not summary:
        summary = raw.strip()
    final = FinalResult(
        summary=summary,
        consensus=consensus,
        divergences=divergences,
        evidences=[e for r in sub_results for e in r.evidences],
    )
    return final, local_usage


def sub_from_dict(d: dict) -> SubResult:
    evs = [
        Evidence(
            metric=e.get("metric", ""),
            formula=e.get("formula", ""),
            sql=e.get("sql", ""),
            window=e.get("window", ""),
            rows=list(e.get("rows") or []),
            row_count=int(e.get("row_count") or 0),
        )
        for e in (d.get("evidences") or [])
    ]
    return SubResult(
        skill_id=d.get("skill_id", ""),
        skill_name=d.get("skill_name", ""),
        conclusion=d.get("conclusion", ""),
        evidences=evs,
        status=d.get("status", "done"),
        note=d.get("note", ""),
    )
