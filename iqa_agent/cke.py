# -*- coding: utf-8 -*-
"""CKE 核心：锚点选择、锦标赛、Bradley-Terry、B-C 分歧、裁判、双门控。

全程零 MOS：驱动信号 = B（分析型）与 C（比较型）两种范式的一致性 + 失真阶梯。
"""
import asyncio
import itertools
import json
import random
import re

import numpy as np

from .client import gather_with_progress
from .prompts.judge import build_judge_prompt
from .prompts.pairwise import build_pairwise_prompt
from .prompts.skills import SKILL_ORDER, build_skill_prompt
from .scoring import parse_score

RULE_LINE_RE = re.compile(r"^\[([A-Z]+)\]\s*(.+)$")


# ---------- 锚点选择（B_score 十分位分层） ----------

def select_anchors(workset_rows: list[dict], k: int = 50) -> list[dict]:
    valid = [r for r in workset_rows if r.get("score") is not None]
    valid.sort(key=lambda r: r["score"])
    n = len(valid)
    anchors = []
    per_bin = max(1, k // 10)
    for b in range(10):
        lo, hi = int(n * b / 10), int(n * (b + 1) / 10)
        bin_rows = valid[lo:hi]
        if not bin_rows:
            continue
        step = max(1, len(bin_rows) // per_bin)
        anchors.extend(bin_rows[:: step][: per_bin])
    return anchors[:k]


# ---------- 锚点锦标赛 ----------

def _parse_winner(text: str) -> str | None:
    m = re.search(r'"winner"\s*:\s*"([AB])"', text)
    if m:
        return m.group(1)
    m = re.search(r"\b([AB])\b", text)
    return m.group(1) if m else None


async def run_tournament(client, anchors: list[dict], seed: int = 42) -> np.ndarray:
    """全配对 1,225 场，随机左右顺序。返回 wins[i,j] = i 胜 j 次数。"""
    n = len(anchors)
    wins = np.zeros((n, n))
    rng = random.Random(seed)
    prompt = build_pairwise_prompt()

    async def duel(i, j):
        first, second = (i, j) if rng.random() < 0.5 else (j, i)
        text, _ = await client.compare_images(anchors[first]["path"], anchors[second]["path"], prompt)
        w = _parse_winner(text)
        if w == "A":
            return first, second
        if w == "B":
            return second, first
        return None

    pairs = list(itertools.combinations(range(n), 2))
    results = await gather_with_progress([duel(i, j) for i, j in pairs], every=200, label="tournament")
    n_invalid = 0
    for r in results:
        if r is None:
            n_invalid += 1
        else:
            w, l = r
            wins[w, l] += 1
    if n_invalid:
        print(f"  [tournament] 无效比较: {n_invalid}/{len(pairs)}")
    return wins


# ---------- Bradley-Terry（Hunter MM 算法） ----------

def fit_bt(wins: np.ndarray, iters: int = 200) -> np.ndarray:
    n = wins.shape[0]
    w = np.ones(n)
    for _ in range(iters):
        w_new = np.copy(w)
        for i in range(n):
            total = sum(wins[i, j] + wins[j, i] for j in range(n) if j != i)
            if total == 0:
                continue
            denom = sum((wins[i, j] + wins[j, i]) / (w[i] + w[j]) for j in range(n) if j != i)
            w_new[i] = wins[i].sum() / denom if denom > 0 else w[i]
        w_new /= np.exp(np.mean(np.log(w_new + 1e-12)))
        w = w_new
    return np.log(w + 1e-12)


# ---------- B-C 分歧 ----------

def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def bc_disagreement(b_scores: np.ndarray, c_scores: np.ndarray) -> np.ndarray:
    """返回每个锚点的 |z(B) - z(C)|。"""
    return np.abs(zscore(b_scores) - zscore(c_scores))


# ---------- 裁判 ----------

async def run_judge(client, anchors: list[dict], b_z: np.ndarray, c_z: np.ndarray,
                    disagree_idx: list[int], agree_idx: list[int],
                    temperature: float, max_rules: int = 5) -> tuple[str, list[str]]:
    """分批喂图（高分歧 + 高一致对照），返回 (raw_text, 候选规则列表)。"""
    prompt = build_judge_prompt(max_rules)
    picks = disagree_idx + agree_idx
    header_lines = []
    for rank, i in enumerate(picks):
        group = "DISAGREE" if rank < len(disagree_idx) else "AGREE"
        header_lines.append(f"Image {rank + 1} [{group}]: B-rank-z={b_z[i]:+.2f}, C-rank-z={c_z[i]:+.2f}")
    header = "Here are the images with their two-paradigm rankings:\n" + "\n".join(header_lines)

    content = [{"type": "text", "text": prompt + "\n\n" + header}]
    for i in picks:
        with open(anchors[i]["path"], "rb") as f:
            import base64
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    messages = [{"role": "user", "content": content}]
    text, _ = await client.chat(messages, temperature=temperature, max_tokens=1200)
    rules = []
    for line in text.splitlines():
        m = RULE_LINE_RE.match(line.strip())
        if m:
            rules.append(f"[{m.group(1)}] {m.group(2).strip()}")
    return text, rules[:max_rules]


# ---------- 双门控 ----------

def tagged_skills_of(rules: list[str]) -> list[str]:
    tag2skill = {"TECH": "S-TECH", "AESTH": "S-AESTH", "CONTENT": "S-CONTENT",
                 "NATURAL": "S-NATURAL", "GLOBAL": "S-GLOBAL"}
    skills = set()
    for r in rules:
        m = RULE_LINE_RE.match(r.strip())
        if m and m.group(1) in tag2skill:
            skills.add(tag2skill[m.group(1)])
    return sorted(skills) or SKILL_ORDER


async def ladder_monotonicity(client, ladder_items: list[dict], img_dir: str,
                              skills: list[str], rules_by_skill: dict | None,
                              scale) -> float:
    """在阶梯子集上测指定 Skill 的整体单调性（严格递减对比例）。"""
    async def one(item, sk):
        rules = (rules_by_skill or {}).get(sk)
        prompt = build_skill_prompt(sk, "koniq", rules=rules)
        text, _ = await client.score_image(f"{img_dir}/{item['file']}", prompt, temperature=0.0)
        p = parse_score(text, scale)
        return (item["src_idx"], item["family"], item["level"], sk, p["score"] if p else None)

    tasks = [one(it, sk) for it in ladder_items for sk in skills]
    rows = await gather_with_progress(tasks, every=300, label="gate-ladder")
    by = {}
    for src, fam, lv, sk, sc in rows:
        if sc is not None:
            by.setdefault((src, fam, sk), {})[lv] = sc
    ok = tot = 0
    for (src, fam, sk), d in by.items():
        if fam == "orig":
            continue
        orig = by.get((src, "orig", sk), {}).get(0)
        seq = [orig] + [d.get(i) for i in (1, 2, 3)]
        if any(v is None for v in seq):
            continue
        ok += sum(1 for a, b in zip(seq, seq[1:]) if a > b)
        tot += 3
    return ok / tot if tot else 0.0


async def anchor_b_scores(client, anchors: list[dict], rules_by_skill: dict | None,
                          scale_key: str, scale) -> np.ndarray:
    """重测锚点 B 分（注入规则后的融合分），用于 B-C 分歧门控。"""
    from .pipeline import run_r2  # 延迟导入防循环
    rows = await run_r2(client, anchors, scale_key, scale, dynamic=False,
                        rules_by_skill=rules_by_skill)
    return np.array([r["score"] if r["score"] is not None else np.nan for r in rows])
