# -*- coding: utf-8 -*-
"""评分输出解析：三级容错（JSON → 正则 → 全文数字），并钳制到量程。"""
import json
import re


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_score(text: str, scale: tuple[float, float]):
    """从模型输出提取 {level, score, reason}。
    返回 (dict|None)。level ∈ 1..5，score ∈ scale。"""
    lo, hi = scale
    # 一级：JSON（容忍 markdown 代码块包裹）
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            level = int(obj.get("level"))
            score = float(obj.get("score"))
            reason = str(obj.get("reason", "")).strip()
            if 1 <= level <= 5:
                return {"level": level, "score": clamp(score, lo, hi),
                        "reason": reason, "parse_tier": 1}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # 二级：正则抓 level/score 字段
    lm = re.search(r"level[\"'\s:]+([1-5])\b", text)
    sm = re.search(r"score[\"'\s:]+([0-9]+(?:\.[0-9]+)?)", text)
    if lm and sm:
        return {"level": int(lm.group(1)), "score": clamp(float(sm.group(1)), lo, hi),
                "reason": "", "parse_tier": 2}
    # 三级：全文中找分数（优先 score-like 小数）
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for nstr in nums:
        v = float(nstr)
        if lo <= v <= hi and v != int(v) or (lo <= v <= hi and len(nums) == 1):
            level = min(5, max(1, int(round((v - lo) / (hi - lo) * 4)) + 1))
            return {"level": level, "score": clamp(v, lo, hi), "reason": "", "parse_tier": 3}
    return None
