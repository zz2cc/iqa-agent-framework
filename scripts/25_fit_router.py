# -*- coding: utf-8 -*-
"""Router 融合权重拟合（D8）：在失真阶梯上做坐标下降 + 4 族交叉验证。

训练信号：阶梯的自造 Oracle（排序已知），零 MOS。
输入：  runs/ladder/eval_main_*/scores.csv（32B 阶梯评测输出）
输出：  runs/router_weights.json            （全体阶梯上拟合的最终权重）
        runs/router_weights_cv_report.json  （4 族交叉验证报告：拟合 vs 等权）
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.prompts.skills import SKILL_ORDER

FAMILIES = ["blur", "noise", "jpeg", "dark"]


def load_ladder_scores(cfg):
    evals = sorted(glob.glob(os.path.join(cfg.ladder_dir, "eval_main_*")), reverse=True)
    assert evals, "未找到 32B 阶梯评测输出，先跑 scripts/20_ladder_eval.py --model main"
    import csv
    path = os.path.join(evals[0], "scores.csv")
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # data[(src, family, skill)][level] = score
    data = {}
    for r in rows:
        if r["score"] in ("", "None", None):
            continue
        data.setdefault((int(r["src_idx"]), r["family"], r["skill"]), {})[int(r["level"])] = float(r["score"])
    print(f"[fit] 加载 {len(rows)} 行 ← {path}")
    return data


def build_sequences(data, families):
    """构造 (seq_per_skill, family) 列表：每个 (源图, 族) → 各 Skill 的 4 级分数序列。"""
    seqs = []
    src_set = sorted({k[0] for k in data})
    for src in src_set:
        for fam in families:
            per_skill = {}
            ok = True
            for sk in SKILL_ORDER:
                orig = data.get((src, "orig", sk), {}).get(0)
                lv = data.get((src, fam, sk), {})
                seq = [orig, lv.get(1), lv.get(2), lv.get(3)]
                if any(v is None for v in seq):
                    ok = False
                    break
                per_skill[sk] = seq
            if ok:
                seqs.append((per_skill, fam))
    return seqs


def monotonicity_with_weights(seqs, weights, families_filter=None):
    """融合分（加权平均）在序列上的严格递减对比例。"""
    ok = tot = 0
    for per_skill, fam in seqs:
        if families_filter and fam not in families_filter:
            continue
        fused = [sum(weights[sk] * per_skill[sk][i] for sk in SKILL_ORDER) for i in range(4)]
        ok += sum(1 for a, b in zip(fused, fused[1:]) if a > b)
        tot += 3
    return ok / tot if tot else 0.0


def fit_weights(seqs, families_filter=None, start=None, step=0.2, rounds=6):
    w = dict(start) if start else {sk: 1.0 for sk in SKILL_ORDER}
    best = monotonicity_with_weights(seqs, w, families_filter)
    for _ in range(rounds):
        improved = False
        for sk in SKILL_ORDER:
            for delta in (step, -step, step / 2, -step / 2):
                trial = dict(w)
                trial[sk] = max(0.1, trial[sk] + delta)
                score = monotonicity_with_weights(seqs, trial, families_filter)
                if score > best:
                    w, best = trial, score
                    improved = True
        if not improved:
            step /= 2
    return w, best


def main():
    cfg = get_config()
    data = load_ladder_scores(cfg)
    seqs = build_sequences(data, FAMILIES)
    print(f"[fit] 有效序列 {len(seqs)} 条")

    equal = {sk: 1.0 for sk in SKILL_ORDER}
    report = {"equal_weight_cv": {}, "fitted_cv": {}}

    # 4 族交叉验证：3 族训练，1 族验证
    for held in FAMILIES:
        train_fams = [f for f in FAMILIES if f != held]
        w_fit, train_score = fit_weights(seqs, train_fams)
        test_score = monotonicity_with_weights(seqs, w_fit, [held])
        test_equal = monotonicity_with_weights(seqs, equal, [held])
        report["fitted_cv"][held] = {"train": round(train_score, 4), "test": round(test_score, 4)}
        report["equal_weight_cv"][held] = {"test": round(test_equal, 4)}
        print(f"[cv] 留出 {held:<6} 拟合 test={test_score:.3f}  等权 test={test_equal:.3f}")

    # 最终权重：全体阶梯拟合
    w_final, score_final = fit_weights(seqs)
    score_equal_all = monotonicity_with_weights(seqs, equal)
    report["final"] = {"weights": w_final, "mono": round(score_final, 4),
                       "equal_mono": round(score_equal_all, 4)}
    print(f"[final] 拟合权重 {w_final} mono={score_final:.3f}（等权 {score_equal_all:.3f}）")

    with open(os.path.join(cfg.runs_dir, "router_weights.json"), "w") as f:
        json.dump(w_final, f, indent=1)
    with open(os.path.join(cfg.runs_dir, "router_weights_cv_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("[done] runs/router_weights.json + router_weights_cv_report.json")


if __name__ == "__main__":
    main()
