# -*- coding: utf-8 -*-
"""Router 相关 prompt：画像器（分诊台）。

设计约束：只问"毛病类型"，不问分数；输出严格 JSON。
类别与阶梯失真族对齐（blur/noise/jpeg/dark 可直接查敏感度矩阵）。
"""

ISSUE_CATEGORIES = ["blur", "noise", "exposure", "color", "composition", "processing", "content"]

PROFILER_PROMPT = """\
You are a quality-issue triage expert. Look at this image and identify the 1-3 DOMINANT quality issue categories.

Categories:
- blur: out-of-focus softness, motion blur, lack of crisp detail
- noise: visible grain, color speckles, high-ISO noise
- exposure: too dark, too bright, clipped highlights, crushed shadows
- color: unnatural color cast, white-balance problems, clashing colors
- composition: poor framing, awkward subject placement, cluttered background
- processing: over-sharpening halos, HDR glow, oversaturation, plastic smoothing
- content: subject cut off, occluded, or key regions not visible

Rules:
- Choose 1-3 categories that MOST harm this image's quality.
- If the image looks fine, choose the single most relevant category anyway (least-bad issue).
Reply with JSON only: {"issues": ["<category>", ...]}"""


def build_profiler_prompt() -> str:
    return PROFILER_PROMPT
