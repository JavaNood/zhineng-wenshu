"""配置：环境变量、模型白名单、路径。"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根：backend/app/config.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# ---- 路径 ----
DATA_DIR = ROOT / os.getenv("DATA_DIR", "data/CN_Ecommerce_Ops")
ONTOLOGY_DIR = ROOT / "ontology"
SEMANTIC_DIR = ROOT / "semantic"
SKILL_DIR = ROOT / "skills"
RUNTIME_DIR = ROOT / "runtime"
RUNTIME_DIR.mkdir(exist_ok=True)

# ---- LLM ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

# 模型白名单。deepseek-chat 不返回思维链，默认用思考模型 deepseek-reasoner。
MODEL_WHITELIST = ["deepseek-chat", "deepseek-reasoner"]
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-reasoner").strip()
if DEFAULT_MODEL not in MODEL_WHITELIST:
    DEFAULT_MODEL = "deepseek-reasoner"

# ---- 编排约束（硬约束，需求第十节） ----
MAX_PARALLEL = 3          # 子 Agent 最大并行度
MAX_STEPS = 30            # 单个专家按模型回复计，上限 30
SQL_IDLE_NUDGE = 4        # 计划已定、连续这么多轮没有新 SQL，要求下一回复直接查
SQL_IDLE_STOP = 8         # 再没有新 SQL 就停止该专家，避免空转到步数上限
MAX_SUB_AGENTS = 3        # 分析意图最多派几个子 Agent（并行度对齐）
PLAN_SOFT = 6             # 成功做完这么多步就提醒收拢
PLAN_STEPS = 10           # 计划最多多少步；第 10 步做完后只写结论
RESULT_ROWS = 2000        # 查询结果回传和证据存档的同一上限。门店、月份、渠道、门店×月都在这个数以内


def resolve_model(model: str | None) -> str:
    """前端回传模型校验：命中白名单否则回退默认。"""
    if model and model in MODEL_WHITELIST:
        return model
    return DEFAULT_MODEL


def has_key() -> bool:
    return bool(DEEPSEEK_API_KEY)
