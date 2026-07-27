# -*- coding: utf-8 -*-
"""推理管线：R1-bare / R1-rich / R2 / R2.5（R3 = R2.5 + 规则库，S5 阶段接入）。"""
import asyncio
import json
from types import SimpleNamespace

from .client import VLMClient, gather_with_progress
from .data import ImageRef
from .prompts.router import build_profiler_prompt
from .prompts.skills import SKILL_ORDER, build_r1_bare_prompt, build_r1_rich_prompt, build_skill_prompt
from .router import (build_explanation, fuse_trimmed, fuse_weighted,
                     issues_to_skill_weights, select_skills)
from .scoring import parse_score


def _as_ref(img):
    """兼容 ImageRef 与 dict（CKE 锚点是带 img_id/path 键的行记录）。"""
    if isinstance(img, dict):
        return SimpleNamespace(img_id=img["img_id"], path=img["path"],
                               dataset=img.get("dataset", "koniq"))
    return img


async def run_r1(client: VLMClient, images: list[ImageRef], scale_key: str,
                 rich: bool, scale: tuple[float, float]) -> list[dict]:
    """单专家单调用评分。rich=True 用 S-GLOBAL 完整细则，否则一句话裸问。"""
    images = [_as_ref(i) for i in images]
    prompt = build_r1_rich_prompt(scale_key) if rich else build_r1_bare_prompt(scale_key)

    async def one(img: ImageRef):
        text, usage = await client.score_image(img.path, prompt, temperature=0.0)
        parsed = parse_score(text, scale)
        row = {"img_id": img.img_id, "dataset": img.dataset, "route": "r1r" if rich else "r1b"}
        if parsed:
            row.update({"level": parsed["level"], "score": parsed["score"],
                        "reason": parsed["reason"], "parse_tier": parsed["parse_tier"]})
        else:
            row.update({"level": None, "score": None, "reason": text[:200], "parse_tier": 0})
        return row

    tasks = [one(img) for img in images]
    results = await gather_with_progress(tasks, every=100, label="r1r" if rich else "r1b")
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            results[i] = {"img_id": images[i].img_id, "dataset": images[i].dataset,
                          "route": "r1r" if rich else "r1b", "level": None, "score": None,
                          "reason": f"API_ERROR: {r}"[:200], "parse_tier": 0}
    return results


def _parse_issues(text: str) -> list[str]:
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        issues = obj.get("issues", [])
        return [i for i in issues if isinstance(i, str)][:3] or ["blur"]
    except (json.JSONDecodeError, ValueError):
        return ["blur"]  # 解析失败兜底：最常见问题类


async def run_r2(client: VLMClient, images: list[ImageRef], scale_key: str,
                 scale: tuple[float, float], dynamic: bool = False,
                 sensitivity: dict | None = None, fitted: dict | None = None,
                 rules_by_skill: dict[str, list[str]] | None = None) -> list[dict]:
    """多专家 + Router 融合。
    dynamic=False → R2：全激活 + trimmed mean；
    dynamic=True  → R2.5：画像分诊 + top-3 激活 + 拟合权重（缺省时回退 trimmed）。
    rules_by_skill：R3 用，tag 定向注入的规则库。
    """
    images = [_as_ref(i) for i in images]

    async def one(img: ImageRef):
        # ① 维度选择
        if dynamic:
            ptext, _ = await client.score_image(img.path, build_profiler_prompt(), temperature=0.0)
            issues = _parse_issues(ptext)
            sw = issues_to_skill_weights(issues, sensitivity)
            active = select_skills(sw, top_k=3)
        else:
            issues, active = [], SKILL_ORDER

        # ② 各 Skill 评分（可注入规则库）
        async def score_skill(sk):
            rules = (rules_by_skill or {}).get(sk)
            prompt = build_skill_prompt(sk, scale_key, rules=rules)
            text, _ = await client.score_image(img.path, prompt, temperature=0.0)
            return sk, parse_score(text, scale)

        results = await asyncio.gather(*[score_skill(sk) for sk in active])
        per_skill = {sk: p for sk, p in results if p}
        scores = {sk: p["score"] for sk, p in per_skill.items()}
        if not scores:
            return {"img_id": img.img_id, "dataset": img.dataset,
                    "route": "r25" if dynamic else "r2", "score": None, "parse_tier": 0}

        # ③ 冲突裁决 + 解释生成
        if dynamic and fitted:
            final, eff_w = fuse_weighted(scores, fitted)
        else:
            final, eff_w = fuse_trimmed(scores)
        explanation = build_explanation(per_skill, eff_w)
        return {
            "img_id": img.img_id, "dataset": img.dataset, "route": "r25" if dynamic else "r2",
            "level": None, "score": round(final, 4),
            "issues": "|".join(issues), "active_skills": "|".join(active),
            "skill_scores": json.dumps(scores, ensure_ascii=False),
            "reason": explanation, "parse_tier": 1,
        }

    tasks = [one(img) for img in images]
    results = await gather_with_progress(tasks, every=100, label="r2.5" if dynamic else "r2")
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            results[i] = {"img_id": images[i].img_id, "dataset": images[i].dataset,
                          "route": "r25" if dynamic else "r2", "score": None,
                          "reason": f"API_ERROR: {r}"[:200], "parse_tier": 0}
    return results
