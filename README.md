# IQA Agent：基于开源多模态大模型的图像质量评估 Agent 框架

> **任务**：在不训练 Backbone 大模型的前提下，通过 Skill (Prompt) 与受限的 Router/Decision 层，让模型直接输出与人群主观评分（MOS）对齐的图像质量分数。
>
> **成绩**：KonIQ-10k Val SRCC **0.734** / MAE 0.491 · SPAQ Test SRCC **0.893** / MAE 0.945

---

## 目录

- [1. 框架架构](#1-框架架构)
- [2. 环境配置](#2-环境配置)
- [3. 快速复现](#3-快速复现)
- [4. 目录结构](#4-目录结构)
- [5. 实验流程（完整重跑）](#5-实验流程完整重跑)
- [6. 缓存机制](#6-缓存机制)
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

| 层 | 说明 | 训练 |
|---|---|---|
| Backbone | `qwen3-vl-32b-instruct`（DashScope API，T=0） | 禁止（§4.1） |
| Skill 技能库 | 3 个评估专家（S-TECH/S-GLOBAL/S-CONTENT），各含 7 个可独立开关的提示词组件 | 不可训练 |
| Router/Decision | 门控矩阵 W(3×7)（21 个浮点数）、冲突裁决、解释生成 | **唯一可训练层**（§4.3） |
| 监督信号 | 合成失真阶梯 + 两两比较 BT 锦标赛（训练集像素自衍生，零 MOS 接触） | — |

## 2. 环境配置

```bash
# 1. 安装 Python 包
pip install numpy scipy pillow
# 如果要用 --api 模式调大模型打分，额外安装：
pip install openai

# 2. 配置环境
cp .env.example .env
# 编辑 .env，必填以下三项：
```

`.env` 中需要：

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 数据集路径（必填！如果数据集不在项目根目录的 "评测数据集/"）：
IQADATA=D:/path/to/datasets
```

**数据集目录结构**（从任务书百度网盘下载后解压，放在项目根目录 `评测数据集/` 或通过 `IQADATA` 指向任意位置）：

```
评测数据集/
├── koniq-10k/
│   ├── 512x384/            # KonIQ 验证集图片（*.jpg）
│   ├── koniq10k_val.csv    # 验证集标签
│   └── koniq10k_train.csv  # 训练集标签（仅用于离线特征构造）
└── SPAQ/
    ├── images/TestImage/   # SPAQ 测试集图片（*.jpg）
    └── spaqTest.csv        # 测试集标签
```

---

## 3. 快速复现

配好 `.env` 和数据集后，一行验证。

### 3.1 默认模式：零 API，验证全部指标

```bash
python scripts/verify.py
```

脚本自动完成四项检查，全程不调 API，约 10 分钟（大部分时间在逐张算 OpenCV 特征）。

**输出内容：**

**[1/5] 环境自检** — Python 版本、包依赖、数据集路径、`.env` API Key、必要冻结文件。

**[2/5] 主表** — R1-bare 至 R6 共 6 臂 + R1-anchor v3 + 8B R1-bare，双数据集全部指标。每行标注数据来源目录。8 臂 × 2 数据集 × 2 指标 = 32 个数字共同构成主表。

**[3/5] R6 统一框架管线** — 逐张图计算 7 维像素特征，读冻结的门控矩阵 W(3×7)，走完整链路：特征 → softmax 话语权 → 融合分 → 与裸问投票混合 → 最终分。前 3 张图打印每一步的中间值（特征值、logit、话语权、融合分、投票分）。最后对比管线重算结果与已缓存的 scores.csv——两者应对齐。

**[4/5] CKE 规则展示** — 9 条自进化规则全文。内部指标全改善，外部 SRCC 仅 +0.001——在最终框架中已移除。

### 3.2 API 抽样模式：验证整条 API 链路可复现

```bash
python scripts/verify.py --api
```

在默认模式基础上，取 KonIQ 和 SPAQ 各 200 张，**真正调用 32B 模型**重跑 S-TECH / S-GLOBAL / S-CONTENT 三个专家的评分，然后走与默认模式相同的门控融合管线。对比 API 评分与缓存专家分的差异、API 管线最终分与缓存 R6 分的差异。

预计 API 调用：200×(3+3) ≈ 1,200 次（KonIQ 600 + SPAQ 600），约 ¥8-15。

### 3.3 HTML 报告

```bash
python scripts/verify.py --html        # 零 API + HTML
python scripts/verify.py --api --html  # API 抽样 + HTML
```

输出 `verify_report.html`，浏览器打开即看。含主表 + 散点图 + 门控矩阵热力图。

---

## 4. 目录结构

```
.
├── iqa_agent/                # 核心框架库
│   ├── config.py             #   全局配置（IQADATA 环境变量支持外部数据集）
│   ├── client.py             #   VLM 客户端（API调用+缓存+并发）
│   ├── data.py               #   数据加载 — 全项目唯一 MOS 读取入口 load_mos()
│   ├── router.py             #   Router/Decision 层（特征提取/门控/融合）
│   ├── scoring.py            #   分数解析（JSON→数值）
│   ├── metrics.py            #   评测指标（SRCC / MAE / PLCC）
│   ├── pipeline.py           #   实验管线（run_r1 / run_r2 / run_r6）
│   └── prompts/              #   全部提示词模板
│       ├── skills.py         #     专家技能 + 裸问 prompt + BARE_PARAS
│       ├── pairwise.py       #     两两比较 prompt（BT 锦标赛用）
│       ├── judge.py          #     裁判 LLM prompt（CKE 用）
│       └── router.py         #     画像器 prompt
├── scripts/                  # 实验脚本
│   ├── verify.py             #   ** 复现唯一入口 **
│   ├── 10_gen_ladder.py      #   合成失真阶梯（开发阶段已完成，无需重跑）
│   ├── 20_ladder_eval.py     #   阶梯体检
│   ├── 25_fit_router.py      #   Router 权重拟合
│   ├── 30_run.py             #   R1~R3 五臂评分
│   ├── 40_cke.py             #   CKE 自进化规则库
│   ├── 45_retro_rescue.py    #   CKE 修复脚本
│   ├── 50_eval.py            #   读 MOS 生成主表（全项目仅此脚本读 MOS）
│   ├── 60_pilot_spaq.py      #   SPAQ 预研
│   ├── 70_figures.py         #   第一轮分析图
│   ├── 80_gen_ladder2.py     #   大梯子 2.0（开发阶段已完成）
│   ├── 80_exam_day.sh        #   考试日一键运行
│   ├── 85_bt_pilot.py        #   BT 试赛（开发阶段已完成）
│   ├── 86_full_tournament.py #   全量 BT 锦标赛
│   ├── 92_expand_tournament.py # 锦标赛扩容（开发阶段已完成）
│   ├── 93_train_router_v3.py #   训练门控矩阵 W(3×7)
│   ├── 94_barevote_pilot.py  #   裸问释义投票 pilot
│   ├── 95_run_r6.py          #   R6 统一框架执行
│   ├── 98_unified_round.py   #   统一框架 alpha 扫描
│   ├── 99_final.py / 99_finish_unified.py  # 统一框架收官
│   ├── apply_32b_w.py        #   32B W → 8B 跨模型迁移
│   ├── r6_hybrid_r3.py       #   R3 经验规则混合 R6（零 API）
│   ├── run_8b_probe.py       #   8B R1-bare + R1-anchor v3
│   ├── run_8b_r6.py          #   8B R6 统一框架
│   ├── make_paper.py         #   生成验收论文（含 8 图嵌入）
│   └── report_figs.py        #   生成全部 8 张图
├── runs/                     # 实验结果（冻结产物）
│   ├── cache/                #   SHA256 磁盘缓存（不入 git）
│   ├── final/                #   主表 + 各臂 scores.csv
│   ├── posthoc/              #   考后诊断臂（R1-anchor v3 / 8B 探针）
│   ├── full_tournament/      #   BT 锦标赛产物
│   ├── router_v3/            #   门控矩阵 fusion_koniq.json（21 参数冻结产物）
│   ├── cke/                  #   CKE 规则库（9 条）
│   ├── ladder/               #   失真阶梯 v1
│   └── ladder2/              #   失真阶梯 v2
├── docs/                     # 文档与交付物
│   ├── 验收论文.docx         #   最终论文（8 图嵌入）
│   ├── 验收论文.md           #   同内容 Markdown 版
│   ├── 汇报_swiss.pptx       #   答辩 PPT（Swiss Minimal 风格，11 页）
│   ├── findings.md           #   研究发现日志 F-001~F-025
│   ├── 验收汇报报告.md       #   验收口径报告
│   └── figs/                 #   全部 8 张图表 PNG
├── 可选任务1.md               # 任务书原文
├── .env.example              # 环境变量模板
└── README.md                 # 本文件
```

> **注意**：`评测数据集/`（KonIQ/SPAQ 图片）和 `runs/cache/`（磁盘缓存）均不入 git。数据集从任务书链接下载，缓存由脚本首次运行时自动生成。

---

## 5. 实验流程（完整重跑）

以下是开发阶段的实际执行顺序。以下步骤的冻结产物（scores.csv、门控矩阵、BT 排行榜）已全部包含在仓库中——**正常复现只需运行 `python scripts/verify.py`**，无需重跑以下脚本。

### 5.1 第一阶段：五臂消融（R1-bare → R1-rich → R2 → R2.5 → R3）

> 产出文件已在 `runs/final/` 中，正常复现无需重跑。

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 合成失真阶梯 | `10_gen_ladder.py` | 纯本地，产出 `runs/ladder/` |
| 阶梯体检 | `20_ladder_eval.py` | 敏感度矩阵，**需调 API** |
| Router 拟合 | `25_fit_router.py` | 纯 numpy，交叉验证 |
| 五臂评分 | `30_run.py` | **大量 API**（约 4.7 万次） |
| CKE 规则库 | `40_cke.py` | **需调 API**（裁判 LLM），约 ¥69 |

CKE 自进化规则库 9 条规则全文可在 `python scripts/verify.py` 的 [4/5] 段查看。内部指标（阶梯单调性、B-C 分歧）全部改善，外部 SRCC 仅 +0.001——已在最终框架中移除。

### 5.2 第二阶段：BT 锦标赛与统一框架 R6

> 以下步骤的冻结产物已在仓库中（BT 排行榜在 `runs/full_tournament/`，门控矩阵在 `runs/router_v3/`），正常复现无需重跑。

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 大梯子 2.0 | `80_gen_ladder2.py` | 扩展失真族，体检用 |
| BT 试赛 | `85_bt_pilot.py` | 门控 S0 验证，**需调 API** |
| 全量锦标赛 | `86_full_tournament.py` | 3.2 万场对决，**大量 API** |
| 锦标赛扩容 | `92_expand_tournament.py` | 2500 节点，**大量 API** |
| 训练门控矩阵 | `93_train_router_v3.py` | 纯 CPU，800 步 SGD，数秒 |
| bare 释义 pilot | `94_barevote_pilot.py` | α 扫描（KonIQ 0.6 / SPAQ 0.3） |
| R6 执行 | `95_run_r6.py` | 专家分复用缓存 + 释义投票新调 |
| 读 MOS | `50_eval.py` | 生成最终主表 |

### 5.3 8B 对照与跨模型迁移

```bash
python scripts/run_8b_probe.py    # 8B R1-bare + R1-anchor v3（需调 API）
python scripts/run_8b_r6.py       # 8B R6 统一框架（需调 API）
python scripts/apply_32b_w.py     # 32B W → 8B（零新增 API，交叉验证）
python scripts/r6_hybrid_r3.py    # R3 经验规则混合验证（零 API）
```

### 5.4 生成论文与图表

```bash
python scripts/report_figs.py     # → docs/figs/*.png
python scripts/make_paper.py      # → docs/验收论文.docx
```

---

## 6. 缓存机制

### 6.1 缓存原理

每次 API 调用经 `VLMClient.score_image()`（`iqa_agent/client.py`），自动 SHA256 磁盘缓存：

```
缓存 Key = SHA256(model_name + prompt_text + image_path + temperature)
缓存文件 = runs/cache/{前2位}/{完整hex}.json
```

### 6.2 缓存的行为

- 首次 API 调用返回后，完整响应（文本 + tokens 用量）写入缓存
- 后续相同参数的调用直接返回已保存的响应，不再发出网络请求——**在当前这份代码和数据上，只需跑一次**
- T=0 的确定性使缓存在同一环境内可靠复用

### 6.3 缓存的边界

缓存保存的是**某次具体的 API 响应**，不是"标准答案"：

- API 服务端可能静默更新模型权重，导致同一 prompt 的输出变化
- 不同 API Key 的账户配置可能影响响应
- 因此 `runs/cache/` 不入 git——其他研究者应在自己的环境中通过重跑脚本生成自己的缓存

### 6.4 跨实验缓存复用

- **R6 统一框架**复用了第一轮的三个专家分——只重新融合、不重新打分，大部分零新增 API
- **8B 跨模型迁移**（`apply_32b_w.py`）通过 8B 独立缓存命名空间，32B 与 8B 缓存互不干扰
- 账本对账：`client.ledger()` 输出 `api_calls` 和 `cache_hits`——零 API 调用即全缓存命中

---

## 7. 合规声明

| 边界 | 执行情况 |
|---|---|
| Backbone 零训练零微调（§4.1） | 全程 qwen3-vl-32b-instruct API 调用，T=0，零参数更新 |
| 训练仅 Router/Decision 层（§4.3） | 门控矩阵 W(3×7) 的 21 个参数为唯一被训练对象 |
| 评测集不用于任何训练（§4.1） | KonIQ Val / SPAQ Test 图像仅在跑分时被模型看到 |
| §4.4 禁止信息不作为在线输入 | 数据集名/Image ID/MOS/失真类型等级/划分信息——全项目零出现 |
| MOS 仅离线评测/误差分析（§4.5） | `load_mos()` 为全项目唯一 MOS 读取入口（`iqa_agent/data.py`） |
| 训练集像素使用批准 | KonIQ/SPAQ Train 像素用于自监督信号构造，经指导教师书批准 |
| 监督信号全部自衍生 | 合成失真阶梯 + BT 锦标赛——零 MOS 接触 |
| 结果经复现验证 | 从 GitHub 裸 clone 后运行 `python scripts/verify.py`，全部指标与原始实验精确一致 |
