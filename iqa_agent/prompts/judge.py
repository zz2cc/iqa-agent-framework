# -*- coding: utf-8 -*-
"""CKE 裁判 prompt（两步分析法，借鉴 MemRefine JUDGE_USER_TEMPLATE）。

裁判看到的输入（由 cke.py 组装）：
  - 一批图像（高分歧组 + 高一致组）
  - 每张图的 B 分（分析型范式）与 C 分（比较型范式）排名
输出：≤5 条带 tag 的抽象规则，domain-agnostic。
"""

JUDGE_PROMPT = """\
You are a senior image-quality-assessment coach. The same VLM judged these images under TWO paradigms:
- B (analytic): scored by a panel of dimension experts, then fused.
- C (comparative): ranked by pairwise "which is better" duels.

Group 1 (HIGH DISAGREEMENT): B and C strongly disagree on these images.
Group 2 (HIGH AGREEMENT): B and C agree closely.

Your task, in two steps:

STEP 1 — Concrete analysis (for yourself, be specific):
For each Group-1 image, describe what you actually see (distortion type, severity, content)
and hypothesize WHY the analytic panel misjudged it relative to the duel-based ranking.
Contrast with Group-2: what makes them easy to judge consistently?

STEP 2 — Abstract rules (this is what gets saved):
Write AT MOST {max_rules} calibration rules that would make the analytic panel better.
Requirements for each rule:
- Prefix with exactly one tag: [TECH] [AESTH] [CONTENT] [NATURAL] [GLOBAL] [GENERAL]
- Abstract and general: a reader must NOT be able to tell which specific images inspired it.
- Actionable: phrased as something an assessor can apply while scoring (e.g., "When ..., do not ...; instead ...").
- One sentence each.

Output format (strict):
ANALYSIS: <3-6 sentences>
RULES:
[TAG] rule one
[TAG] rule two
..."""


def build_judge_prompt(max_rules: int = 5) -> str:
    return JUDGE_PROMPT.format(max_rules=max_rules)
