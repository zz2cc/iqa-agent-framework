# -*- coding: utf-8 -*-
"""C 路线（比较型）两两对比 prompt。

每对随机左右顺序（抗位置偏置），由 cke.py 控制 A/B 摆放。
"""
PAIRWISE_PROMPT = """\
Two images are shown: the FIRST is image A, the SECOND is image B.
Judge which one has better OVERALL technical quality (sharpness, noise, exposure, artifacts, naturalness).
Ignore subject preference and composition style; focus on fidelity and visual cleanliness.
Reply with JSON only: {"winner": "A"} or {"winner": "B"}"""


def build_pairwise_prompt() -> str:
    return PAIRWISE_PROMPT
