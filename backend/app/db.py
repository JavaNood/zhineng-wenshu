"""DuckDB 连接与视图注册：内存库，启动时把 7 个 CSV 注册为英文别名视图。"""
from __future__ import annotations
import duckdb
from .config import DATA_DIR

# 中文表名 -> 英文别名（SQL 中建议用英文别名，避免引号负担）
ALIASES = {
    "fact_订单": ("orders", "fact_订单.csv"),
    "fact_订单明细": ("items", "fact_订单明细.csv"),
    "fact_月度目标": ("target", "fact_月度目标.csv"),
    "dim_日期": ("d_date", "dim_日期.csv"),
    "dim_客户": ("d_cust", "dim_客户.csv"),
    "dim_门店": ("d_store", "dim_门店.csv"),
    "dim_商品": ("d_prod", "dim_商品.csv"),
}

_conn: duckdb.DuckDBPyConnection | None = None


def init_db() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is not None:
        return _conn
    _conn = duckdb.connect(":memory:")
    for _cn, (alias, fname) in ALIASES.items():
        _conn.execute(
            f'CREATE OR REPLACE VIEW {alias} AS '
            f"SELECT * FROM read_csv_auto('{DATA_DIR / fname}', header=true)"
        )
    return _conn


def get_connection() -> duckdb.DuckDBPyConnection:
    if _conn is None:
        return init_db()
    return _conn


def new_cursor() -> duckdb.DuckDBPyConnection:
    """每个子 Agent 独立 cursor（DuckDB 连接非线程安全）。"""
    return get_connection().cursor()


# 给 LLM 的表别名与关键字段说明（写进工具提示，避免猜表名）
ALIAS_DOC = {
    "orders": "fact_订单（订单级，金额/渠道/状态）",
    "items": "fact_订单明细（订单商品行，类目结构分析）",
    "target": "fact_月度目标（门店月度销售/毛利/订单量目标）",
    "d_date": "dim_日期（年/季度/月/年月/促销标记）",
    "d_cust": "dim_客户（客户等级/企业客户）",
    "d_store": "dim_门店（大区/城市/渠道类型）",
    "d_prod": "dim_商品（一/二/三级类目/品牌）",
}
