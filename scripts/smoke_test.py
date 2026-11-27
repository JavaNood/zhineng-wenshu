"""结构/连通性冒烟测试：不调 LLM，验证本体、技能库、DuckDB 取数、应用可导入。
对齐 D5：不写死标准答案，只校验结构与连通。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db import init_db, new_cursor
from app.tools import sql_tool
from app.ontology import loader as onto
from app.semantic import loader as sem
from app.skills import loader as skilllib
from app.skills import files as skillfiles
from app.agents.main_react import reject_dispatch
from app.agents.fusion import title_problem, validate_fuse, validate_rename


def _session_skill_roundtrip() -> None:
    assert title_problem("量价拆解与季节节奏归因专家")
    assert title_problem("量价与季节归因专家-小王")
    assert title_problem("量价与季节归因专家-小吴") == ""
    session_id = "smoke-session"
    try:
        skillfiles.write_session_skill(
            session_id, "fuse-1",
            title="量价与季节融合专家-小周",
            description="把量价和季节放在一条步骤里。",
            trigger="销售额变化需要同时看结构和季节",
            expert_role="融合分析",
            instructions="1. 先看季节。\n2. 再做量价闭合。",
            output_schema="季节判断；量价分解",
            validation="先季节后量价",
            parents=["value-price-breakdown", "seasonality-diagnosis"],
            ancestors=["value-price-breakdown", "seasonality-diagnosis"],
        )
        base_ids = {item["id"] for item in skilllib.list_skills()}
        assert "fuse-1" not in base_ids
        scoped = {item["id"]: item for item in skilllib.list_skills(session_id)}
        assert "fuse-1" in scoped
        assert "本会话用户融合的专家" in scoped["fuse-1"]["description"]
        assert skilllib.load_skills().get("fuse-1") is None
        card = skilllib.get_skill("fuse-1", session_id)
        assert card and card["scope"] == "session"
        assert card["ancestors"] == ["value-price-breakdown", "seasonality-diagnosis"]
        skillfiles.set_title(session_id, "fuse-1", "自定义名字")
        assert skilllib.get_skill("fuse-1", session_id)["title"] == "自定义名字"
        assert reject_dispatch(["fuse-1"], session_id) is None
        assert reject_dispatch(["fuse-1", "value-price-breakdown"], session_id)
        assert reject_dispatch(["value-price-breakdown"], session_id)
        assert reject_dispatch(
            ["value-price-breakdown", "customer-refund"], session_id) is None
        options = [
            {"kind": "expert", "skill_id": "value-price-breakdown"},
            {"kind": "expert", "skill_id": "seasonality-diagnosis"},
            {"kind": "text"},
        ]
        assert validate_fuse(["value-price-breakdown"], "", options)
        assert validate_fuse(
            ["value-price-breakdown", "seasonality-diagnosis"], "只要季节", options)
        assert validate_fuse(
            ["value-price-breakdown", "seasonality-diagnosis"], "", options) is None
        assert validate_fuse([], "只保留季节对照", options) is None
        rename_options = [
            {"key": "A", "kind": "choice"},
            {"key": "B", "kind": "text"},
        ]
        assert validate_rename(["A"], "", rename_options) is None
        assert validate_rename(["B"], "", rename_options)
        assert validate_rename(["B"], "新名字", rename_options) is None
        assert validate_rename(["A"], "新名字", rename_options)
        print("[会话技能] 隔离、改名、派工互斥通过")
    finally:
        skillfiles.delete_session_dir(session_id)


def main():
    # 1) 本体校验
    problems = onto.validate_ontology()
    print("[本体校验]", "通过" if not problems else f"问题: {problems}")
    assert not problems, problems

    # 2) 指标对齐
    m = sem.resolve_certified_metric("销售额")
    print("[指标对齐 销售额]", m["name"], "=", m["expression"])
    assert m and m["code"] == "sales_amount"
    m2 = sem.resolve_certified_metric("客单价")
    print("[指标对齐 客单价]", m2["name"], "=", m2.get("formula") or m2.get("expression"))

    # 3) 技能库加载：基础专家不包含会话目录
    skills = skilllib.list_skills()
    print("[技能库]", [s["id"] for s in skills])
    assert len(skills) == 6
    names = {s["name"] for s in skills}
    assert "量价结构拆解专家-小王" in names
    assert "时序季节诊断专家-小黄" in names
    assert "目标与门店对标专家-小李" in names
    assert "客群与退款诊断专家-小陈" in names
    assert "毛利成本结构专家-小周" in names
    assert "商品与品牌结构专家-小赵" in names
    _session_skill_roundtrip()

    # 4) DuckDB 真实取数（只读）
    init_db()
    cur = new_cursor()
    r = sql_tool.run_sql(
        "SELECT SUM(实付金额)-SUM(退款金额) AS sales, COUNT(DISTINCT 订单ID) AS orders FROM orders",
        cursor=cur)
    print("[销售额总量]", r["rows"], "rows=", r["row_count"], "err=", r["error"])
    assert not r["error"] and r["row_count"] == 1

    # 5) 维度 join：渠道环比（2025Q4 vs Q3 示意）
    r2 = sql_tool.run_sql(
        """SELECT o.销售渠道,
           SUM(CASE WHEN d.年月 BETWEEN '2025-10' AND '2025-12' THEN o.实付金额-o.退款金额 ELSE 0 END) AS q4,
           SUM(CASE WHEN d.年月 BETWEEN '2025-07' AND '2025-09' THEN o.实付金额-o.退款金额 ELSE 0 END) AS q3
           FROM orders o JOIN d_date d ON o.订单日期=d.日期
           GROUP BY 1 ORDER BY q4 DESC""",
        cursor=cur)
    print("[渠道Q4/Q3]", r2["rows"][:3], "err=", r2["error"])
    assert not r2["error"]

    # 6) 只读拦截
    bad = sql_tool.run_sql("DROP TABLE orders")
    print("[只读拦截]", bad["error"])
    assert bad["error"]

    # 7) 应用可导入
    import app.main  # noqa
    print("[应用导入] OK")
    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
