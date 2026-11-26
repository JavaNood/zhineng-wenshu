"""工具目录。在线执行走 batch.classify，这里只保留说明。"""
from __future__ import annotations

TOOL_SPECS = [
    {"name": "read_entity", "layer": "ontology", "desc": "按实体名读取属性和标识。"},
    {"name": "read_value_domain", "layer": "ontology", "desc": "按值域名读取枚举或取值来源。"},
    {"name": "read_relations", "layer": "ontology", "desc": "按实体读取允许的关系。"},
    {"name": "read_policies", "layer": "ontology", "desc": "按主题读取策略正文。"},
    {"name": "read_logical_dataset", "layer": "semantic", "desc": "读取逻辑数据集的列、类型和粒度。"},
    {"name": "resolve_certified_metric", "layer": "semantic", "desc": "把自然语言指标对齐认证指标。"},
    {"name": "read_sql_plan", "layer": "semantic", "desc": "读取指标的 SQL 规划约束，不返回可执行 SQL。"},
    {"name": "describe_schema", "layer": "physical", "desc": "读取视图别名、来源文件、列名和类型。"},
    {"name": "list_skills", "layer": "orchestrator", "desc": "列出分析专家。"},
]
