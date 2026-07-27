# IQA Agent：基于开源多模态大模型的图像质量评估 Agent 框架

> **任务**：在不训练 Backbone 大模型的前提下，通过 Skill (Prompt) 与受限的 Router/Decision 层，让模型直接输出与人群主观评分（MOS）对齐的图像质量分数。
>
> **成绩**：KonIQ-10k Val SRCC **0.734** / MAE 0.491 · SPAQ Test SRCC **0.893** / MAE 0.945

---

## 目录

- [1. 框架架构](#1-框架架构)
- [2. 环境配置](#2-环境配置)
- [3. 目录结构](#3-目录结构)
- [4. 实验复现](#4-实验复现)
- [5. 缓存机制](#5-缓存机制)
- [6. 关键脚本速查](#6-关键脚本速查)
- [7. 合规声明](#7-合规声明)

---

## 1. 框架架构

```
输入图片
  ├─ 路一：技能专家并行评分（S-TECH / S-GLOBAL / S-CONTENT）
  ├─ 路二：像素统计特征（7 维 OpenCV 手工特征）
  └─ 路三：裸问释义投票（4 条同义问法取均值）
        ↓
  门控矩阵 W(3×7)：根据特征动态分配专家话语权
        ↓
  融合分 = g · s（话语权 × 专家分）
        ↓
  最终分 = α · 融合分 + (1−α) · 投票分
```

| 层 | 说明 | 是否可训练 |
|---|---|---|
| Backbone | `qwen3-vl-32b-instruct`（DashScope API，T=0） | 禁止（§4.1） |
| Skill 技能库 | 3 个评估专家，各含 7 个可独立开关的提示词组件 | 不可训练 |
| Router/Decision | 门控矩阵 W(3×7)、冲突裁决、解释生成 | **唯一可训练层**（§4.3） |
| 监督信号 | 合成失真阶梯 + 两两比较 BT 锦标赛（训练集像素自衍生，零 MOS） | — |

---

## 2. 环境配置

```bash
# Python 环境
pip install openai numpy scipy pillow python-docx matplotlib

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY=sk-xxxxxxxx
```

`.env` 中需要两个变量：

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

---

## 3. 目录结构

```
.
├── iqa_agent/                # 核心框架库
│   ├── config.py             #   全局配置（路径/标尺/模型名）
│   ├── client.py             #   VLM 客户端（API调用+缓存+并发）
│   ├── data.py               #   数据加载 — 全项目唯一 MOS 读取入口 load_mos()
│   ├── router.py             #   Router/Decision 层（特征提取/门控/融合/解释）
│   ├── scoring.py            #   分数解析（parse_score：自由文本→数值）
│   ├── metrics.py            #   评测指标（SRCC / MAE / PLCC）
│   ├── pipeline.py           #   实验管线（run_r1 / run_r2 / run_r6）
│   ├── cke.py                #   CKE 自进化规则库（第一轮R3用）
│   ├── experience.py         #   经验规则注入逻辑
│   └── prompts/              #   全部提示词模板
│       ├── skills.py         #     3 个专家技能 + 裸问 prompt
│       ├── pairwise.py       #     两两比较 prompt（BT 锦标赛用）
│       ├── judge.py          #     裁判 LLM prompt（CKE 用）
│       └── router.py         #     画像器 prompt
├── scripts/                  # 实验脚本（按流水线编号 10→99）
│   ├── 10_gen_ladder.py      #   生成合成失真阶梯（纯本地）
│   ├── 20_ladder_eval.py     #   阶梯体检 + 敏感度矩阵
│   ├── 25_fit_router.py      #   Router 融合权重拟合（第一轮）
│   ├── 30_run.py             #   R1 / R2 / R2.5 / R3 评分执行
│   ├── 40_cke.py             #   CKE 四轮迭代
│   ├── 45_retro_rescue.py    #   CKE 回溯修复
│   ├── 50_eval.py            #   读 MOS 生成主表（全项目仅此脚本读 MOS）
│   ├── 60_pilot_spaq.py      #   SPAQ 预研
│   ├── 70_figures.py         #   第一轮分析图
│   ├── 80_gen_ladder2.py     #   大梯子 2.0
│   ├── 80_exam_day.sh        #   考试日一键运行（第一轮五臂）
│   ├── 85_bt_pilot.py        #   BT 锦标赛试赛（S0，门控验证）
│   ├── 86_full_tournament.py #   全量 BT 锦标赛（KonIQ 32,389 场 / SPAQ 4,045 场）
│   ├── 92_expand_tournament.py # 锦标赛扩容
│   ├── 93_train_router_v3.py #   训练门控矩阵 W(3×7)（R6 核心）
│   ├── 94_barevote_pilot.py  #   裸问释义投票 pilot
│   ├── 95_run_r6.py          #   R6 统一框架执行
│   ├── 96_anchor_probe.py    #   R1-anchor 锚点归因探针（考后 F-024）
│   ├── 97_r6_offanchor.py    #   R6-offanchor 去锚点消融（考后 F-025）
│   ├── 98_unified_round.py   #   统一框架 alpha 扫描
│   ├── 99_final.py           #   统一框架收官
│   ├── 99_finish_unified.py  #   统一框架收官续篇
│   ├── apply_32b_w.py        #   32B 门控矩阵迁移至 8B（跨模型迁移）
│   ├── r6_hybrid_r3.py       #   R3 经验规则混合 R6（零 API 纯融合实验）
│   ├── run_8b_probe.py       #   8B R1-bare + R1-anchor 探针
│   ├── run_8b_r6.py          #   8B R6 统一框架
│   ├── make_paper.py         #   生成验收论文（.docx + .md，8 图嵌入）
│   ├── report_figs.py        #   生成全部 8 张 dataviz 规范图表
│   └── smoke_test.py         #   冒烟测试
├── runs/                     # 实验结果（缓存 + 冻结产物）
│   ├── cache/                #   SHA256 磁盘缓存（不入 git）
│   ├── final/                #   主表 + 各臂 scores.csv + 分析图
│   ├── posthoc/              #   考后诊断臂产物（R1-anchor / R6-offanchor / 8B 探针）
│   ├── bt_pilot/             #   BT 试赛产物
│   ├── full_tournament/      #   全量 BT 锦标赛产物（对决记录 + 排行榜）
│   ├── router_v3/            #   门控矩阵 fusion_koniq.json（21 参数，BT 训练冻结产物）
│   ├── cke/                  #   CKE 规则库（9 条）
│   ├── ladder/               #   失真阶梯 v1
│   └── ladder2/              #   失真阶梯 v2
├── docs/                     # 文档与交付物
│   ├── 验收论文.docx         #   最终论文（8 图嵌入，896KB）
│   ├── 验收论文.md           #   同内容 Markdown 版
│   ├── 汇报_swiss.pptx       #   5 分钟答辩 PPT（Swiss Minimal 风格，11 页）
│   ├── findings.md           #   研究发现日志 F-001~F-025
│   ├── 验收汇报报告.md       #   验收口径报告（三轮方法/主表/困难分析）
│   ├── 项目运行情况与结果分析报告.md  # 全程运行报告
│   └── figs/                 #   全部 8 张图表 PNG
├── 可选任务1.md               # 任务书原文
├── .env.example              # 环境变量模板
└── README.md                 # 本文件
```

> **注意**：`评测数据集/`（KonIQ/SPAQ 图片）和 `runs/cache/`（SHA256 磁盘缓存）均不入 git。数据集从任务书链接下载，缓存由脚本首次运行时自动生成。

---

## 4. 实验复现

核心流程分两个阶段：**第一轮五臂消融**（R1-bare → R1-rich → R2 → R2.5 → R3）→ **BT 锦标赛 + 统一框架**（R6）。脚本按编号 10→99 顺序执行。重跑已完成的步骤会自动命中缓存。

### 4.1 第一轮：五臂消融

> 目标：逐层检验 Prompt 工程、多专家融合、Router 优化、自监督迭代各自的净收益。

```bash
# Step 1: 生成合成失真阶梯（纯本地）
python scripts/10_gen_ladder.py

# Step 2: 阶梯体检 — 测每个专家对每类失真的敏感度
python scripts/20_ladder_eval.py

# Step 3: Router 融合权重拟合 + 4 族交叉验证
python scripts/25_fit_router.py

# Step 4: 五臂评分（R1b / R1r / R2 / R2.5 / R3，双域，约 4.7 万次 API 调用）
python scripts/30_run.py
# 或使用考试日一键脚本：
bash scripts/80_exam_day.sh

# Step 5: CKE 自进化规则库（B/C 分歧→裁判提炼→双门控筛选）
python scripts/40_cke.py --round 1

# Step 6: 第一次读 MOS，生成首轮主表
python scripts/50_eval.py --runs runs/final
```

### 4.2 第二阶段：BT 锦标赛与统一框架 R6

> 目标：以两两比较锦标赛为自监督信号，训练门控矩阵，完成统一框架。

```bash
# Step 7: 大梯子 2.0（扩展失真族，涵盖 SPAQ）
python scripts/80_gen_ladder2.py

# Step 8: BT 锦标赛试赛（小规模，门控 S0 验证）
python scripts/85_bt_pilot.py

# Step 9: 全量 BT 锦标赛（KonIQ 2,500 节点 32,389 场 / SPAQ 400 节点 4,045 场）
python scripts/86_full_tournament.py --domain koniq
python scripts/86_full_tournament.py --domain spaq

# Step 10: 锦标赛扩容（监督信号增强）
python scripts/92_expand_tournament.py

# Step 11: 训练逐图动态门控矩阵 W(3×7)
#     输入：7 维特征 + BT 强度分（监督）
#     输出：runs/router_v3/fusion_koniq.json（21 个冻结参数）
python scripts/93_train_router_v3.py

# Step 12: bare 释义投票 pilot（α 扫描，KonIQ α=0.6 / SPAQ α=0.3）
python scripts/94_barevote_pilot.py

# Step 13: R6 统一框架执行（专家分复用缓存，仅释义投票为新 API 调用）
python scripts/95_run_r6.py

# Step 14: 第二次（末次）读 MOS，生成最终主表
python scripts/50_eval.py --runs runs/final
```

### 4.3 考后诊断与跨模型验证

```bash
# R1-anchor 锚点归因探针（F-024）：分解 R1-bare→R1-rich 的增益来源
python scripts/96_anchor_probe.py

# R6-offanchor 去锚点消融（F-025）：验证锚点在多专家融合中的角色
python scripts/97_r6_offanchor.py

# 8B 对比：R1-bare + R1-anchor v3（双域）
python scripts/run_8b_probe.py

# 8B R6 统一框架（三专家+释义投票）
python scripts/run_8b_r6.py

# 跨模型迁移：32B 训练的门控矩阵 W → 8B 专家分（零新增 API）
python scripts/apply_32b_w.py

# R3 经验规则混合验证（纯本地，零 API）：证明 CKE 规则无益
python scripts/r6_hybrid_r3.py
```

### 4.4 生成论文与图表

```bash
# 生成全部 8 张 dataviz 规范图表 → docs/figs/
python scripts/report_figs.py

# 生成验收论文（含 8 图嵌入）→ docs/验收论文.docx + docs/验收论文.md
python scripts/make_paper.py
```

---

## 5. 缓存机制

### 5.1 缓存原理

每次 API 调用都经过 `VLMClient.score_image()`（`iqa_agent/client.py`），调用时自动做 SHA256 磁盘缓存：

```
缓存 Key = SHA256(model_name + prompt_text + image_path + temperature)
缓存文件 = runs/cache/{前2位}/{完整hex}.json
```

### 5.2 为什么缓存是可靠的

- **温度 = 0** 意味着同一 (模型, prompt, 图片) 组合每次调用结果完全相同——缓存是"答案的永久副本"
- 首次调用写入缓存；后续相同参数调用**秒级命中**，**零 API 消耗**、零费用
- 缓存文件内容为完整的 API 响应（含 tokens 用量、生成文本）

### 5.3 跨实验缓存复用

这是本项目的关键工程特性：

- **第一轮 R1/R2** 的 5 专家评分全部缓存
- **R6 统一框架** 复用了第一轮的三个专家分——只重新融合、不重新打分，**大部分零新增 API**
- **8B 跨模型迁移**（`apply_32b_w.py`）通过 `VLMClient(cfg, cfg.model_debug)` 自动使用 8B 的独立缓存命名空间，32B 与 8B 缓存互不干扰
- **账本对账**：`client.ledger()` 输出 `{"api_calls": N, "cache_hits": M}`——`api_calls=0` 表示全部来自缓存

### 5.4 缓存文件不入 git

`runs/cache/` 在 `.gitignore` 中排除，因为：
1. 文件数量极大（数万个小 JSON）
2. 其他研究者可在自己的环境中通过重跑脚本自动生成
3. 不同 API Key 的响应理论上可不同（实际 T=0 差异极小）

如需跨机器迁移缓存，直接复制 `runs/cache/` 目录——缓存 Key 不含机器/路径信息。

---

## 6. 关键脚本速查

| 脚本 | 功能 | 调 API | 产出 |
|---|---|---|---|
| `10_gen_ladder.py` | 合成失真阶梯 | 否 | `runs/ladder/` |
| `30_run.py` | R1/R2/R2.5/R3 五臂评分 | **是** | `runs/final/r1b_*/`, `r2_*/`, `r3_*/` |
| `40_cke.py` | CKE 四轮自进化 | **是** | `runs/cke/` |
| `50_eval.py` | **全项目唯一 MOS 读取入口** | 否 | 打印 SRCC/MAE/PLCC |
| `86_full_tournament.py` | BT 全量锦标赛 | **是** | `runs/full_tournament/` |
| `93_train_router_v3.py` | 训练逐图动态门控 W(3×7) | 否 | `runs/router_v3/fusion_koniq.json` |
| `95_run_r6.py` | R6 统一框架执行 | **是**（仅释义投票） | `runs/final/r6_*/` |
| `96_anchor_probe.py` | 锚点归因探针 | **是** | `runs/posthoc/r1anchor_*/` |
| `97_r6_offanchor.py` | 去锚点消融 | **是** | `runs/posthoc/r6offanchor_*/` |
| `apply_32b_w.py` | 32B W → 8B 跨模型迁移 | 否（全缓存命中） | 打印指标 |
| `run_8b_probe.py` | 8B R1-bare + R1-anchor 探针 | **是** | `runs/posthoc/r1b_8b_*/`, `r1anchor_v3_*/` |
| `run_8b_r6.py` | 8B R6 统一框架 | **是** | `runs/posthoc/r6_8b_*/` |
| `make_paper.py` | 生成验收论文（含 8 图嵌入） | 否 | `docs/验收论文.docx` |
| `report_figs.py` | 生成全部 8 张 dataviz 图 | 否 | `docs/figs/*.png` |

---

## 7. 合规声明

| 边界 | 执行情况 |
|---|---|
| Backbone 零训练零微调（§4.1） | 全程 qwen3-vl-32b-instruct API 调用，T=0，零参数更新 |
| 训练仅 Router/Decision 层（§4.3） | 门控矩阵 W(3×7) 的 21 个参数为唯一被训练对象 |
| 评测集不用于任何训练（§4.1） | KonIQ Val / SPAQ Test 图像仅在跑分时被模型"看"到，不参与任何训练回路 |
| §4.4 禁止信息不作为在线输入 | 数据集名 / Image ID / MOS / 失真类型等级 / 划分信息——全项目在 Router/LLM/Skill 中零出现 |
| MOS 仅离线评测/误差分析（§4.5） | `load_mos()` 为全项目唯一 MOS 读取入口（`iqa_agent/data.py`）；两轮各读一次 + 考后诊断臂一次 |
| 训练集像素使用批准 | KonIQ/SPAQ Train 像素用于自监督信号构造，经指导教师 2026-07-23 邮件书面批准 |
| 监督信号全部自衍生 | 合成失真阶梯 + BT 锦标赛——零 MOS 接触 |
| 结果可复现 | T=0 + SHA256 磁盘缓存 + 种子固定 → 任意臂一键复现 |
