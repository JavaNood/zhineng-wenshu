"""临时探查：统计真实枚举值、日期范围、口径闭合。仅开发期使用。"""
import duckdb, os

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "CN_Ecommerce_Ops")
con = duckdb.connect(":memory:")

def view(name, file):
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{os.path.join(DATA, file)}', header=true, all_varchar=false)")

view("orders", "fact_订单.csv")
view("items", "fact_订单明细.csv")
view("target", "fact_月度目标.csv")
view("d_date", "dim_日期.csv")
view("d_cust", "dim_客户.csv")
view("d_store", "dim_门店.csv")
view("d_prod", "dim_商品.csv")

def q(sql):
    return con.execute(sql).fetchall()

print("== 日期范围 ==")
print("订单日期:", q("SELECT min(订单日期), max(订单日期) FROM orders"))
print("目标年月:", q("SELECT min(年月), max(年月) FROM target"))

for col, tbl, label in [
    ("销售渠道", "orders", "销售渠道"),
    ("订单状态", "orders", "订单状态"),
    ("支付方式", "orders", "支付方式"),
    ("是否新客首单", "orders", "是否新客首单"),
    ("客户等级", "d_cust", "客户等级"),
    ("是否企业客户", "d_cust", "是否企业客户"),
    ("性别", "d_cust", "性别"),
    ("大区", "d_store", "大区"),
    ("渠道类型", "d_store", "渠道类型"),
    ("促销标记", "d_date", "促销标记"),
    ("一级类目", "d_prod", "一级类目"),
]:
    rows = q(f"SELECT {col}, COUNT(*) FROM {tbl} GROUP BY 1 ORDER BY 2 DESC")
    print(f"\n== {label} ==")
    for r in rows:
        print("  ", r)

print("\n== 口径闭合校验（全量）==")
print("实付合计:", q("SELECT SUM(实付金额) FROM orders"))
print("退款合计:", q("SELECT SUM(退款金额) FROM orders"))
print("销售额=实付-退款:", q("SELECT SUM(实付金额)-SUM(退款金额) FROM orders"))
print("订单量(distinct 订单ID):", q("SELECT COUNT(DISTINCT 订单ID) FROM orders"))
print("毛利额合计:", q("SELECT SUM(毛利额) FROM orders"))
print("商品金额/折扣金额:", q("SELECT SUM(商品金额), SUM(折扣金额) FROM orders"))
print("目标销售合计:", q("SELECT SUM(销售目标) FROM target"))
