# 智能问数 · 多专家并行分析 Demo

基于开源零售数据集（7 个 CSV），演示两件事：本体层与业务语义层分开治理，以及主从 Agent 按官方函数调用查数、派专家、给出可回溯证据。

用户用一句自然语言提问。主 Agent 自己决定查知识、写 SQL、先报告数值、再派专家。分析专家先写计划，再按同一套查数工具取数。每条结论可以回看 SQL 和结果。

> 数据来源：https://github.com/hccccc01333/powerbi-retail-analysis （只用 CN_Ecommerce_Ops 的 7 个 CSV）
> 实现说明见 `docs/开发文档.md`。

## 一、快速启动

### 1. 后端

```bash
cd 智能问数
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-...

PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

后端同时提供 API。若已执行 `npm run build`，同一端口也托管 `frontend/dist`。

### 2. 前端

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173 ，/api 代理到 8000
```

生产构建：`npm run build`，产物在 `frontend/dist/`。

### 3. 自检

不调用真实模型：

```bash
PYTHONPATH=backend ./.venv/bin/python scripts/smoke_test.py
PYTHONPATH=backend ./.venv/bin/python scripts/graph_smoke.py
```

`smoke_test.py` 校验两层知识、认证指标、六位专家、DuckDB 只读拦截。`graph_smoke.py` 用假模型校验主图路由、专家并行、步数上限和官方 `tool_calls` 协议。

## 二、页面上能看到什么

- 左上角标题下方是「本体」开关，字号与「智能问数」相同。默认开启。这一轮还在回答时不能切换，切换只影响之后发出的问题。
- 还没提问时，欢迎语中间显示当前状态：「已开启本体知识访问」或「已关闭本体知识访问能力，Agent回答可信度降低」。
- 回答结束后不自动把这句状态放在该轮末尾。只有用户拨动开关，才出现在这一轮末尾。下一轮不自动继承。
- 本体关闭的那一轮，气泡下方有红字「本轮回答未使用本体知识」。本体开着的轮次没有这句。
- 每一轮结束显示 `输入 N tokens · 输出 N tokens`。下一行是「你还可以问我：」，三个问题各一类。分析轮等融合取消或改名完成后再出现；改名完成后开头是「专家已经生成好了」。点击问题只填进输入框。再提下一问后，上一轮的数字还在。
- 模型下拉在右上角，白名单是 `deepseek-chat` 与 `deepseek-reasoner`，默认 `deepseek-reasoner`。
- 灰色「思考」来自 `reasoning_content`。黑色步骤是模型写给用户的 `content`。工具名和参数走官方 `tool_calls`，页面上是单独的工具卡片。`finish` 只在服务端收口，不画成工具卡片。寒暄的回复是正文，不把 `finish` 的内部说明再印一遍。

## 三、演示问题

- 问数：`2025年的销售额是多少？`
- 问数：`2025年各渠道的销售额和订单量`
- 分析：`2025年1季度和2季度的销售额是多少，变化的原因是什么？`

本体开启时，写 SQL 之前要先成功调用 `resolve_certified_metric` 和 `read_sql_plan`。本体关闭时，这两层工具不可用，先 `describe_schema` 再由模型自己写 SQL。数字是否不同不作为通过条件。

只问数值时，`report` 成功后直接结束。原因类问题先把点名的数值 `report` 给用户，再 `list_skills`，再 `dispatch`。SQL、报告和派工不放在同一次回复里。

## 四、目录

```
智能问数/
├── data/CN_Ecommerce_Ops/          # 7 个 CSV
├── ontology/                       # 本体层：实体、值域、关系、策略。不含聚合公式
│   ├── entities.yaml
│   ├── value_domains.yaml
│   ├── relations.yaml
│   └── policies.yaml
├── semantic/                       # 业务语义层：逻辑数据集、认证指标、SQL 规划约束
│   ├── logical_datasets.yaml
│   ├── certified_metrics.yaml
│   └── sql_planner.yaml
├── skills/                         # 六位分析专家，目录 id 不变；会话融合在 skills/_session/
│   ├── value-price-breakdown/      # 量价结构拆解专家-小王
│   ├── seasonality-diagnosis/      # 时序季节诊断专家-小黄
│   ├── benchmark-target/           # 目标与门店对标专家-小李
│   ├── customer-refund/            # 客群与退款诊断专家-小陈
│   ├── margin-cost/                # 毛利成本结构专家-小周
│   └── product-brand/              # 商品与品牌结构专家-小赵
├── backend/app/
│   ├── main.py                     # FastAPI、SSE、静态页
│   ├── llm.py                      # DeepSeek 流式：reasoning_content / content / tool_calls
│   ├── db.py                       # DuckDB 视图别名
│   ├── ontology/loader.py
│   ├── semantic/loader.py
│   ├── tools/                      # batch.py 批处理门闩，schemas.py 官方函数定义
│   └── agents/                     # 在线主图 orchestrator → main_react；专家 plan_execute
├── frontend/                       # React + Vite + TS
├── scripts/smoke_test.py
├── scripts/graph_smoke.py
└── docs/开发文档.md
```

DuckDB 视图：`orders`、`items`、`target`、`d_date`、`d_cust`、`d_store`、`d_prod`。

## 五、认证口径

口径只存在 `semantic/certified_metrics.yaml`。提示词不预写公式。本体开启时，模型通过 `resolve_certified_metric` 读取。

- 销售额 `sales_amount` = `SUM(实付金额) - SUM(退款金额)`
- 订单量 `order_count` = `COUNT(DISTINCT 订单ID)`
- 毛利额 `gross_profit` = `SUM(毛利额)`
- 折扣金额 `discount_amount` = `SUM(折扣金额)`
- 毛利率 = 毛利额 / 销售额
- 客单价 = 销售额 / 订单量
- 折扣率 = `SUM(折扣金额) / SUM(商品金额)`
- 销售目标达成率 = 销售额 / 销售目标
- 毛利目标达成率 = 毛利额 / 毛利目标

`sql_planner.yaml` 只写来源表、时间列、允许的关系、扇出注意。不写现成的 `SELECT`。

## 六、硬约束

- 一次模型回复最多 4 个函数。超过的不执行，并以 `role: tool` 返回「未执行」和原因。主循环按回复次数计，上限 32 次。
- 子 Agent 并行不超过 3。单个专家按回复次数计，上限 30。计划控制在 6 步以内，最多 10 步。一步计划一次成功查询。做完 6 步提醒收拢，第 10 步后只写结论。计划还没做完时，连续 8 轮没有新 SQL 就停止该专家，改计划不算新 SQL，不伪造结论。超限停在已有证据上，不补写完成。谁先做完，卡片就先标成已完成。
- SQL 只允许只读 `SELECT`。写操作在工具层拒绝。
- 查询和报告只覆盖用户点名的指标、时间和维度。原因、变化、诊断交给专家，主 Agent 不用额外 SQL 自己拆。
- 同一页面里连续提问共用一个 `session_id`。融合专家只留在这个会话的 `skills/_session/<session_id>/`。刷新或关闭页面时通知服务端删除。不把上一问的结论带回模型。
- 不预设业务数字，不用正则把某次演示答案写死。
