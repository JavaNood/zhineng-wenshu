"""临时探测：喂模拟专家结论调 merge_results，打印原始返回与 JSON 解析结果。"""
import asyncio, sys
sys.path.insert(0, "backend")
from app.agents.supervisor import merge_results
from app.schemas import SubResult

MOCK = [
    SubResult(skill_id="value-price-breakdown", skill_name="量价结构拆解专家", status="done",
              conclusion="2025销售额微增0.12%，本质量增价减：订单量+162单贡献+33.8万，客单价-17.8元拖累-29.7万，几乎抵消；服饰食品低客单占比上升、美妆数码高客单下降。"),
    SubResult(skill_id="customer-refund", skill_name="客群与退款诊断专家", status="done",
              conclusion="普通客群以量补价失败、单均走弱是最大拖累，中高价值客群正增长部分对冲；新客拉新未走弱，老客尤其普通老客是流失来源。"),
    SubResult(skill_id="benchmark-target", skill_name="目标与门店对标专家", status="done",
              conclusion="线下门店与天猫旗舰店缺口最大，华南大区拖累、华东拉动；部分旗舰/体验店未达目标，属结构换挡而非总量衰退。"),
]

async def main():
    final, usage = await merge_results(MOCK, "deepseek-chat")
    print("=== summary ===")
    print(repr(final.summary[:600]))
    print("=== consensus ===", len(final.consensus))
    for c in final.consensus:
        print(" -", c.get("point", "")[:120], "|", c.get("support"))
    print("=== divergences ===", len(final.divergences))
    for d in final.divergences:
        print(" -", d.get("point", "")[:120])
    print("usage", usage)

asyncio.run(main())
