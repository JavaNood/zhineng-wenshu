# 智能问数 · 多专家并行分析 Demo

基于开源零售数据集（7 个 CSV），演示**指标语义层（本体）**与 **Agent + Skill 多专家并行、结果可比较可验证**两条核心成果。
用户一句自然语言提问，系统完成意图理解 → Plan 编排 → 多 Skill（专家）子 Agent 并行取数 → 共识/分歧汇聚，每条结论绑定可回溯证据（指标口径、SQL、结果快照）。

> 数据来源：https://github.com/hccccc01333/powerbi-retail-analysis （仅用 CN_Ecommerce_Ops 7 个 CSV）

## 一、快速启动

### 1. 后端
```bash
cd 智能问数
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# 配置 API Key（必填，不要写明文进仓库）
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-...

PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. 前端（开发模式）
```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173 ，已代理 /api 到 8000
```
生产构建：`npm run build`，产物在 `frontend/dist/`。

### 3. 连通性自检（不需要 Key）
```bash
./.venv/bin/python scripts/smoke_test.py
```
校验本体、指标对齐、4 个 Skill、DuckDB 真实取数、只读拦截、应用可导入。

## 二、演示话术（现场输入）
- 问数：`2025 年销售额是多少？环比 2024 年如何？`
- 分析：`2025Q4 销售额环比 Q3 下降，原因是什么？`（系统动态选 ≥2 个专家并行）
- 组合：`2025 销售额如何？环比如何？为什么变动了？`（拆成 2 问数点 + 1 分析点）

页面可看：意图拆分与 Plan、子 Agent 并行进度（并排）、各专家结论、共识/分歧、汇总、可展开的 SQL/结果证据、Tokens 统计；右上角下拉切换模型（deepseek-chat / deepseek-reasoner）。

## 三、目录结构
```
智能问数/
├── data/CN_Ecommerce_Ops/      # 7 个 CSV（不入库、不改动）
├── ontology/retail_semantic.yaml   # 指标语义层本体（表/指标口径/维度/关系/枚举）
├── skills/                    # 分析专家 Skill 库（官方 SKILL.md 形态，一专家一目录）
│   ├── value-price-breakdown/SKILL.md    # 量价结构拆解
│   ├── seasonality-diagnosis/SKILL.md    # 时序季节诊断
│   ├── benchmark-target/SKILL.md         # 目标与门店对标
│   └── customer-refund/SKILL.md           # 客群与退款诊断
├── backend/app/
│   ├── config.py / llm.py / db.py / schemas.py / main.py
│   ├── ontology/loader.py          # 本体加载、resolve_metric、校验
│   ├── skills/loader.py            # 技能库加载（解析 */SKILL.md frontmatter+正文）
│   ├── tools/                     # registry / ontology_tools / sql_tool / skill_tools
│   ├── agents/                     # planner / supervisor / worker
│   └── store/session.py            # 会话 JSON 快照
├── frontend/ (React + Vite + TS)
├── scripts/                     # probe_data.py / smoke_test.py
└── docs/开发文档.md
```

## 四、核心指标口径（本体固化，禁止 LLM 自造）
- 销售额 = SUM(实付金额) − SUM(退款金额)
- 订单量 = COUNT(DISTINCT 订单ID)
- 毛利额 = SUM(毛利额)；毛利率 = 毛利额 / 销售额
- 客单价 = 销售额 / 订单量；折扣率 = SUM(折扣金额)/SUM(商品金额)
- 销售目标达成率 = 销售额 / SUM(销售目标)；毛利目标达成率 = 毛利额 / SUM(毛利目标)

## 五、硬约束
- 子 Agent 最大并行 3；单 Agent step 上限 60（超限中止标注，不伪造完成）。
- 不预设答案、不用正则/固定话术伪造结果；SQL 报错在 step 内修正重试，不降级给假答案。
- 单次对话有效，无跨会话记忆；无登录，Key 由后端 `.env` 注入。
- 无 Key 时接口返回明确错误事件，不崩溃、不编造。

## 六、说明
- 编排为自研轻量 supervisor（asyncio 信号量=3），未引入 LangGraph，预留增强位。
- 模型白名单：deepseek-chat / deepseek-reasoner，前端可切，后端校验。
