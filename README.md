# IQA Agent：基于开源多模态大模型的图像质量评估 Agent 框架

> 任务书：《可选任务1》（可选任务1.md）。Backbone：`qwen3-vl-32b-instruct`（DashScope 开源权重模型，零训练零微调）。
> 三轮实验：一轮（7/23-25）五臂消融；二轮（7/25-26）理论驱动的 Router v2 重设计（预注册）；三轮（7/26）Router v3 收官（预注册）；考后诊断（7/25）两项探针（锚点归因 + 跨专家校准）。
> **最终成绩（SRCC）：KonIQ 0.7335（R6，越过 Tool-IQA 零样本锚点 0.729）/ SPAQ 0.8932（R6）；预注册假设 H1′/H3′ 命中、H2′ 半中。**
> 考后诊断（post-hoc）：R1-anchor（裸分+仅锚点）SPAQ MAE 0.807 全表最优；R6-offanchor 证明锚点表是多专家跨维度校准的必需胶水。

---

## 1. 合规声明（验收首先看这个）

| 边界 | 执行情况 | 证据 |
|---|---|---|
| Backbone 零训练零微调 | ✅ 全程 API 调用，无任何参数更新 | 全部代码 |
| 训练仅 Router/Decision 层（§4.3：规则选择/冲突裁决/解释生成） | ✅ Router v2/v3 权重/门控均为此三职能 | docs/预注册-R4R5.md、docs/预注册-R6.md |
| 评测集（KonIQ Val / SPAQ Test）不用于任何训练 | ✅ 考卷像素只在跑分时被模型"看" | ADR-0003 |
| MOS 仅离线评测/误差分析：三轮终评各读一次（均在预注册冻结后）+ 考后诊断臂（R1-anchor）离线评测一次（§4.5 条款，无训练/选择/调参用途） | ✅ `load_mos` 全项目唯一入口 | `grep -r "load_mos" --include="*.py"` |
| §4.4 禁项（数据集名/ID/文件名/失真标签/划分信息）不作为线上输入 | ✅ prompt 净化清单 | docs/PLAN.md §13 |
| KonIQ/SPAQ **Train** 像素（无 MOS）用于 Router 优化 | ✅ 何老师 2026-07-23 邮件书面批准 | 邮件归档 + ADR-0003 |
| 工具/辅助模型未在评测集上训练 | ✅ 仅用 PIL/OpenCV（手工特征，零训练） | ADR-0001#6 |
| MAE 尺度预先声明、不用 MOS 拟合任何映射 | ✅ 原生尺度直接输出，冻结 | ADR-0001#5 |
| 预注册后跑分 | ✅ R4/R5、R6 定义+假设均冻结在先，之后方读 MOS；考后诊断臂另有一次离线评测读取（§4.5 条款） | docs/预注册-R4R5.md、docs/预注册-R6.md |

合规文档链：`docs/ADR-0001-compliance-baseline.md` → `ADR-0002-backbone-and-router.md` → `ADR-0003-round2-pixel-signals.md` → `预注册-R4R5.md` → `预注册-R6.md`。

## 2. 环境

```bash
pip install openai numpy scipy pillow python-docx matplotlib
cp .env.example .env  # 填入 DASHSCOPE_API_KEY=sk-...
```

## 3. 复现步骤

### 一轮（五臂消融，已冻结）

```bash
python scripts/10_gen_ladder.py            # 合成失真阶梯（本地）
python scripts/20_ladder_eval.py           # 阶梯体检 + 敏感度矩阵
python scripts/25_fit_router.py            # Router 拟合 + 4 族交叉验证
python scripts/30_run.py --route r1b --dataset koniq ...   # 五臂 × 双数据集
python scripts/40_cke.py --round 1         # CKE 四轮
python scripts/50_eval.py --runs runs/final  # 第一次读 MOS（已完成，勿重跑）
```

### 二轮（理论驱动重设计，已冻结）

```bash
python scripts/80_gen_ladder2.py           # 大梯子 2.0（KonIQ+SPAQ Train，本地）
python scripts/81_score_ladder2.py         # 梯子评分 + 体检（F-015/F-016）
python scripts/85_bt_pilot.py              # S0：BT 试赛 + 门控（GO）
python scripts/86_full_tournament.py --domain koniq   # S3：全量锦标赛（双域 GO）
python scripts/86_full_tournament.py --domain spaq
python scripts/87_score_protocols.py --domain koniq   # bare/rich 协议分数
python scripts/88_train_router_v2.py --task fusion --domain koniq  # 融合权重
python scripts/88_train_router_v2.py --task route  --domain spaq   # 协议路由
python scripts/89_pilots.py patch          # R5 组件 pilot（patch PASS）
python scripts/89_pilots.py expect --domain spaq     # para3 PASS / koniq FAIL
python scripts/90_run_r45.py --arm r4      # R4 双域（零 API，缓存重融合）
python scripts/90_run_r45.py --arm r5      # R5-SPAQ（para3+patch）
python scripts/50_eval.py --runs runs/final  # 第二次读 MOS（已完成，勿重跑）
python scripts/91_figures_v2.py            # 二轮归因图（考后）
```

### 三轮（Router v3 收官，已冻结）

```bash
python scripts/92_expand_tournament.py     # 锦标赛扩容 2,500 节点（G2 +0.0995/G3 0.795）
python scripts/93_train_router_v3.py       # 逐图动态融合门控（留出 0.814>静态 0.789）
python scripts/94_barevote_pilot.py        # bare 释义投票 pilot（α=0.6）
python scripts/95_run_r6.py                # R6 双域（预注册 docs/预注册-R6.md 冻结在先）
python scripts/50_eval.py --runs runs/final  # 第三次读 MOS（已完成，勿重跑）
```

### 考后诊断（post-hoc，不参与主线结论）

```bash
python scripts/96_anchor_probe.py          # R1-anchor 臂（裸分+仅锚点，F-024）
python scripts/97_r6_offanchor.py          # R6-offanchor 臂（R6−锚点表，F-025）
# 指标补算：load_mos + compute_metrics（§4.5 离线性能评测条款；id 对齐主线臂）
```

所有 API 调用带 SHA256 磁盘缓存（runs/cache/），断点续跑；重跑自动命中缓存。

## 4. 目录导读

| 路径 | 内容 |
|---|---|
| `docs/验收汇报报告.md` | **验收口径报告**（学术论文格式：方案/三轮主表/困难分析/锚相关性定律） |
| `docs/项目运行情况与结果分析报告.md` | 全程运行报告（一轮五臂+二轮重设计+V3+主表 v3+case study） |
| `docs/findings.md` | 发现日志 F-001~F-025（全部证据链） |
| `docs/预注册-R4R5.md` / `docs/预注册-R6.md` | 二轮/三轮预注册（假设与冻结声明） |
| `docs/二轮重设计计划-通俗版.md`、`docs/三轮V3计划-通俗版.md` | 两轮计划（通俗版） |
| `docs/ADR-000*.md` | 三份合规裁决 |
| `runs/final/main_table.csv` | 主表 v3（冻结；R1-anchor 考后臂见验收报告 §4.1） |
| `runs/final/figures_v2/` | 归因图 5 张 |
| `runs/full_tournament/` | BT 锦标赛全部产物（排行榜/门控报告） |
| `runs/router_v2/`、`runs/router_v3/` | Router 权重与门控 |
| `runs/posthoc/` | 考后诊断产物（R1-anchor 分布与摘要，F-024） |

## 5. 一句话科学结论

**自监督零训练 IQA 的天花板 = 优化信号与真实目标的相关性**（锚相关性定律）——一轮发现（自洽性提升 ≠ 对齐提升），二轮预注册验证（同分布 BT 信号使同一机制 +0.007→+0.062），三轮放大利用（KonIQ 0.577→0.633→0.668→0.7335，越过外部零样本锚点）。考后诊断补完锚点角色全貌：**锚点表在 bare 单打分中纯有害（F-024，R1-anchor），在多专家融合中是跨维度校准的必需胶水（F-025，R6-offanchor）**——同一组件在不同架构深度扮演相反角色，R6 处于局部最优。
