"""结构/连通性冒烟测试：不调 LLM，验证本体、技能库、DuckDB 取数、应用可导入。
对齐 D5：不写死标准答案，只校验结构与连通。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db import init_db, new_cursor
from app.tools import sql_tool
from app.ontology import loader as onto
from app.semantic import loader as sem
from app.skills import loader as skilllib


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

    # 3) 技能库加载
    skills = skilllib.list_skills()
    print("[技能库]", [s["id"] for s in skills])
    assert len(skills) == 4
    names = {s["name"] for s in skills}
    assert "量价结构拆解专家-小王" in names
    assert "时序季节诊断专家-小黄" in names
    assert "目标与门店对标专家-小李" in names
    assert "客群与退款诊断专家-小陈" in names

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
