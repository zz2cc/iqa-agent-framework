# IQA Agent：基于开源多模态大模型的图像质量评估 Agent 框架

> **任务**：在不训练 Backbone 的前提下，通过 Skill (Prompt) 与 Router/Decision 层，让模型直接输出与人群主观评分（MOS）对齐的图像质量分。
>
> **成绩**：KonIQ-10k Val SRCC **0.734** / MAE 0.491 · SPAQ Test SRCC **0.893** / MAE 0.945
>
> **入口**：`python scripts/verify.py`（或双击 `verify.bat`）

---

## 目录

- [1. 给老师的三步复现](#1-给老师的三步复现)
- [2. verify.py 模式速览](#2-verifypy-模式速览)
- [3. 框架架构](#3-框架架构)
- [4. 环境配置](#4-环境配置)
- [5. 目录结构](#5-目录结构)
- [6. test_data 机制](#6-test_data-机制)
- [7. 缓存机制](#7-缓存机制)
- [8. 实验流程（完整重跑）](#8-实验流程完整重跑)
- [9. 合规声明](#9-合规声明)

---

## 1. 给老师的三步复现

```bash
# 0. 下载数据集（百度网盘）
#   链接: https://pan.baidu.com/s/16CeAUEb8SaUHI15JzjNvjg  提取码: 1234
#   解压到项目根目录，形成 评测数据集/koniq-10k/ 和 评测数据集/SPAQ/
#   （解压后目录里的 Annotations/ 和 SPAQ.zip 是原始数据集自带的，代码不读取，不用管）

# 1. clone + 装包
git clone https://github.com/zz2cc/iqa-agent-framework.git
cd iqa-agent-framework
pip install numpy scipy pillow openai

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填 DASHSCOPE_API_KEY=sk-xxxxxxxx

# 3. 双击运行
verify.bat
```

`verify.bat` 跑完依次询问：是否 API 抽样验证？是否生成 HTML 报告？选 y 即可。窗口不闪退。

> **不需要 API 验证？** 直接 `python scripts/verify.py --html-only` 秒出 HTML 报告。
> **没下载完整数据集？** `--api` 和 `--api-only` 自动用仓库内的 `test_data/`（200+200张，63MB），无需数据集。

## 2. verify.py 模式速览

| 命令 | 做什么 | API | 费用 |
|---|---|---|---|
| `python scripts/verify.py` | 环境自检 + 主表(8臂) + R6管线 + CKE规则 | 零 | ¥0 |
| `python scripts/verify.py --api` | 同上 + KonIQ×200 + SPAQ×200 调32B重跑 | ~1,200次 | ¥8-15 |
| `python scripts/verify.py --api-only` | 仅API抽样(跳过默认4步，需先跑过默认) | ~1,200次 | ¥8-15 |
| `python scripts/verify.py --html-only` | 仅生成HTML(需先跑过默认) | 零 | ¥0 |

**输出内容：**

- **[1/5] 环境自检** — Python版本、包依赖、数据集路径、API Key、冻结文件完整性
- **[2/5] 主表** — R1-bare 至 R6 + R1-anchor v3 + 8B R1-bare，双域全部指标，标注数据来源
- **[3/5] R6 统一框架管线** — 逐张计算7维特征→W门控→话语权→融合→混合(前3张打印全链路)
- **[4/5] CKE 规则展示** — 9条自进化规则全文(内部全改善，外部+0.001，已被最终框架移除)
- **[5/5] API 抽样** (`--api`时) — 调用32B重跑，与缓存指标对比，排除偶然性
- **对照验证表** (`--api`后) — 缓存 vs API 实时重跑的SRCC/MAE对照，含差值行
- **HTML 报告** (`--html`后) — 自包含HTML，主表+散点图+W热力图+API散点图(如有)，`start verify_report.html`

## 3. 框架架构

```
输入图片
  ├─ 路一：技能专家并行评分（S-TECH / S-GLOBAL / S-CONTENT）
  ├─ 路二：像素统计特征（7维 OpenCV 手工特征）
  └─ 路三：裸问释义投票（4条同义问法取均值）
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
| Skill 技能库 | S-TECH/S-GLOBAL/S-CONTENT，各含7个可独立开关的提示词组件 | 不可训练 |
| Router/Decision | 门控矩阵 W(3×7)（21个浮点数）、冲突裁决、解释生成 | **唯一可训练层**（§4.3） |
| 监督信号 | Bradley-Terry 锦标赛（训练集像素自衍生，零MOS，用于训练唯一可训练的门控矩阵W） | — |
| 7维像素特征 | lap_var(锐度)/noise(噪声)/colorful(色彩)/bright(亮度)/logpix(分辨率)/aspect(宽高比)/spread(专家分歧度) | 纯OpenCV，零训练 |

## 4. 环境配置

```bash
pip install numpy scipy pillow openai
cp .env.example .env
# 编辑 .env，填入 API Key：
```

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

**数据集**（百度网盘）：https://pan.baidu.com/s/16CeAUEb8SaUHI15JzjNvjg  提取码: 1234

下载后解压到项目根目录，形成 `评测数据集/koniq-10k/512x384/` 和 `评测数据集/SPAQ/images/TestImage/`。解压后目录里的 `Annotations/` 和 `SPAQ.zip` 是原始数据集自带的，代码不读取，不用管。

**如果数据集放在其他位置**，在 `.env` 中设置：

```
IQADATA=D:/你的路径/评测数据集
```

程序会从 `$IQADATA/koniq-10k/512x384/` 等子路径读取。不设则默认读项目根目录下的 `评测数据集/`。

> **没有完整数据集也能跑**：仓库内的 `test_data/` 含 KonIQ+SPAQ 各 200 张（63MB）。`--api` 模式在完整数据集缺失时自动回退到 `test_data/`。主表验证需要完整数据集的 MOS 标签 CSV（三个文件共 538KB，已在仓库根目录）。

## 5. 目录结构

```
.
├── verify.bat                 # Windows 一键双击
├── iqa_agent/                 # 核心框架库
│   ├── config.py              #   全局配置（IQADATA 环境变量）
│   ├── client.py              #   VLM 客户端（API调用+缓存+并发）
│   ├── data.py                #   数据加载 — 全项目唯一 MOS 读取入口
│   ├── router.py              #   Router 层（opencv_features / gate_weights / softmax）
│   ├── scoring.py             #   分数解析（JSON/文本→数值）
│   ├── metrics.py             #   评测指标（SRCC/MAE/PLCC）
│   ├── pipeline.py            #   实验管线
│   └── prompts/               #   全部提示词模板
├── scripts/
│   ├── verify.py              #   ** 复现唯一入口 **
│   ├── 10_gen_ladder.py       #   合成失真阶梯（无需重跑）
│   ├── 20_ladder_eval.py      #   阶梯体检
│   ├── 25_fit_router.py       #   Router 权重拟合
│   ├── 30_run.py              #   R1~R3 五臂评分
│   ├── 40_cke.py              #   CKE 自进化（零收益组件，无需重跑）
│   ├── 50_eval.py             #   读 MOS 生成主表
│   ├── 80_gen_ladder2.py      #   大梯子2.0（无需重跑）
│   ├── 85_bt_pilot.py         #   BT 试赛（无需重跑）
│   ├── 86_full_tournament.py  #   全量 BT 锦标赛
│   ├── 92_expand_tournament.py # 锦标赛扩容（无需重跑）
│   ├── 93_train_router_v3.py  #   训练门控矩阵 W(3×7)
│   ├── 94_barevote_pilot.py   #   释义投票 pilot
│   ├── 95_run_r6.py           #   R6 统一框架执行
│   ├── apply_32b_w.py         #   32B W → 8B 跨模型迁移
│   ├── run_8b_probe.py        #   8B 探针
│   ├── run_8b_r6.py           #   8B R6
│   ├── make_paper.py          #   生成验收论文
│   └── report_figs.py         #   生成8张图
├── test_data/                 # API抽样用测试集（KonIQ+SPAQ各200张，63MB）
├── runs/                      # 冻结产物
│   ├── final/                 #   主表 + 各臂 scores.csv
│   ├── posthoc/               #   考后诊断臂
│   ├── full_tournament/       #   BT 锦标赛产物
│   ├── router_v3/             #   门控矩阵 W (21参数)
│   └── cke/                   #   CKE 规则库 (9条)
├── docs/                      # 文档与交付物
│   ├── 验收论文.docx / .md    #   最终论文(8图嵌入)
│   ├── 汇报_swiss.pptx        #   答辩PPT
│   ├── findings.md            #   研究发现日志 F-001~F-025
│   └── figs/                  #   8张图表PNG
├── 可选任务1.md                # 任务书原文
└── .env.example               # 环境变量模板
```

## 6. test_data 机制

仓库内的 `test_data/` 包含 KonIQ 和 SPAQ 各 200 张固定种子(seed=42)抽样的图片和 MOS 标签，共 63MB。

| `.env` 中 IQADATA | `--api` 行为 |
|---|---|
| 已设置且路径有效 | 用完整数据集，固定种子抽样200张 |
| **未设置或路径无效** | 自动回退到 `test_data/`（无需下载30GB数据集） |

verify.py 启动时自动检测：完整数据集图片目录存在 → 用完整集；不存在 → 切到 test_data。输出中会打印 `数据源: test_data` 或 `数据源: 原始数据集`。

> `test_data/spaq/` 图片已压缩到 1568px（与 R6 管线处理 SPAQ 原图的方式一致），KonIQ 图片保持原始 512×384 分辨率。

## 7. 缓存机制

### 原理
```
缓存 Key = SHA256(model_name + prompt_text + image_path + temperature)
缓存文件 = runs/cache/{前2位}/{完整hex}.json
```

### 行为
- T=0：同一组合每次结果相同 → 缓存 = 该次 API 响应的永久保存
- 首次调用写入，后续秒级命中，零 API 消耗、零费用
- `runs/cache/` 不入 git —— 不同API Key需在自己的环境里通过重跑生成

### 边界
- 缓存保存的是某次具体的 API 响应，非"标准答案"
- API 服务端可能静默更新模型权重 → 跨环境缓存不保证一致
- 老师首次 clone 后 `runs/cache/` 为空，第一次 `--api` 走真实网络请求

## 8. 实验流程（完整重跑）

以下步骤的冻结产物（scores.csv、门控矩阵、BT 排行榜）已全在仓库中。**正常复现只需 `python scripts/verify.py`。**

### 8.1 第一阶段：五臂消融（R1-bare → R3）

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 合成失真阶梯 | `10_gen_ladder.py` | 纯本地 |
| 阶梯体检 | `20_ladder_eval.py` | 需调API |
| Router 拟合 | `25_fit_router.py` | 纯numpy |
| 五臂评分 | `30_run.py` | 大量API(约4.7万次) |
| CKE 规则库 | `40_cke.py` | 需调API，约¥69 |

> CKE 9条规则在 `verify.py` [4/5]段打印。内部指标全改善，外部SRCC 仅 +0.001——已移除。

### 8.2 第二阶段：BT 锦标赛与 R6

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 大梯子2.0 | `80_gen_ladder2.py` | 无需重跑 |
| BT 试赛 | `85_bt_pilot.py` | 无需重跑 |
| 全量锦标赛 | `86_full_tournament.py` | 3.2万场对决 |
| 锦标赛扩容 | `92_expand_tournament.py` | 无需重跑 |
| 训练门控矩阵 | `93_train_router_v3.py` | 纯CPU，800步SGD |
| 释义投票pilot | `94_barevote_pilot.py` | α扫描 |
| R6 执行 | `95_run_r6.py` | 专家分复用缓存 |
| 读MOS | `50_eval.py` | 生成最终主表 |

### 8.3 8B 对照与跨模型迁移

```bash
python scripts/run_8b_probe.py     # 8B R1-bare + R1-anchor v3
python scripts/run_8b_r6.py        # 8B R6 统一框架
python scripts/apply_32b_w.py      # 32B W → 8B (零新增API)
python scripts/r6_hybrid_r3.py     # R3 经验规则混合验证 (零API)

# 生成文档
python scripts/report_figs.py      # → docs/figs/*.png
python scripts/make_paper.py       # → docs/验收论文.docx
```

## 9. 合规声明

| 边界 | 执行情况 |
|---|---|
| Backbone 零训练（§4.1） | 全程 API 调用，T=0，零参数更新 |
| 仅 Router 层可训练（§4.3） | 门控矩阵 W(3×7) 的21个参数为唯一被训练对象 |
| 评测集不进入训练（§4.1） | KonIQ Val / SPAQ Test 仅跑分时被模型看到 |
| §4.4 禁止信息不入在线输入 | 数据集名/ID/MOS/失真类型/划分信息全项目零出现 |
| MOS 仅离线评测（§4.5） | `load_mos()` 为唯一 MOS 读取入口 |
| 训练集像素使用批准 | 经指导教师书面批准，仅用于自监督信号构造 |
| 监督信号全部自衍生 | BT 锦标赛（训练集像素自衍生，零 MOS）—— 合成失真阶梯仅用于开发阶段体检，未进入最终框架 |
| 经 GitHub clone 验证 | `python scripts/verify.py` 全指标精确一致 |
