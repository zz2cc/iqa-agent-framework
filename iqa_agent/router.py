# -*- coding: utf-8 -*-
"""Router/Decision 层：维度选择 + 冲突裁决 + 解释生成（任务书 §4.3 三项）。

合规说明（ADR-0002 D8）：
- 全部优化信号来自失真阶梯（自造 Oracle），零 MOS；
- 拟合权重由 scripts/25_fit_router.py 离线产出（runs/router_weights.json），
  缺失时自动回退到稳健融合（trimmed mean + 离群降权）。
"""
import json
import os
import statistics

# issue → 先验增益（无阶梯对应族的类别用固定先验；有族的查敏感度矩阵）
ISSUE_PRIOR_GAIN = {
    "composition": {"S-AESTH": 1.0},
    "content": {"S-CONTENT": 1.0},
    "processing": {"S-NATURAL": 1.0},
    "color": {"S-TECH": 0.3, "S-AESTH": 0.5},
    "exposure": {"S-TECH": 0.5, "S-NATURAL": 0.3},
}
# issue → 阶梯失真族（可查敏感度矩阵）
ISSUE_TO_FAMILY = {"blur": "blur", "noise": "noise", "exposure": "dark"}


# ---------- 冲突裁决 ----------

def iqr_adjusted_weights(scores: dict[str, float]) -> dict[str, float]:
    """离群降权：|s_i - median| > 1.5*IQR 的维度权重减半。"""
    vals = list(scores.values())
    n = len(vals)
    weights = {k: 1.0 for k in scores}
    if n < 4:
        return weights
    srt = sorted(vals)
    q1 = statistics.median(srt[: n // 2])
    q3 = statistics.median(srt[(n + 1) // 2:])
    iqr = q3 - q1
    med = statistics.median(vals)
    for k, v in scores.items():
        if abs(v - med) > 1.5 * iqr:
            weights[k] = 0.5
    return weights


def fuse_trimmed(scores: dict[str, float]) -> tuple[float, dict[str, float]]:
    """R2 基线融合：trimmed mean（n≥4 去最高最低）+ 离群降权。"""
    adj = iqr_adjusted_weights(scores)
    items = sorted(scores.items(), key=lambda kv: kv[1])
    if len(items) >= 4:
        items = items[1:-1]
    num = sum(v * adj[k] for k, v in items)
    den = sum(adj[k] for k, v in items)
    return num / den, adj


def fuse_weighted(scores: dict[str, float], fitted: dict[str, float]) -> tuple[float, dict[str, float]]:
    """R2.5 训练后融合：拟合权重 × 离群降权兜底。"""
    adj = iqr_adjusted_weights(scores)
    eff = {k: fitted.get(k, 1.0) * adj[k] for k in scores}
    num = sum(scores[k] * eff[k] for k in scores)
    den = sum(eff.values())
    return (num / den if den > 0 else statistics.mean(scores.values())), eff


# ---------- 维度选择 ----------

def issues_to_skill_weights(issues: list[str], sensitivity: dict | None) -> dict[str, float]:
    """画像类别 → Skill 权重。有阶梯族的查敏感度矩阵，无族的用先验。"""
    w = {s: 1.0 for s in ["S-TECH", "S-AESTH", "S-CONTENT", "S-NATURAL", "S-GLOBAL"]}
    for issue in issues:
        fam = ISSUE_TO_FAMILY.get(issue)
        if fam and sensitivity:
            for skill, fams in sensitivity.items():
                w[skill] = w.get(skill, 1.0) + fams.get(fam, 0.0)
        for skill, gain in ISSUE_PRIOR_GAIN.get(issue, {}).items():
            w[skill] = w.get(skill, 1.0) + gain
    return w


def select_skills(skill_weights: dict[str, float], top_k: int = 3) -> list[str]:
    """取权重最高的 top_k 个 Skill（S-GLOBAL 永远保留作对照锚）。"""
    ranked = sorted(skill_weights, key=skill_weights.get, reverse=True)
    picked = ranked[:top_k]
    if "S-GLOBAL" not in picked:
        picked[-1] = "S-GLOBAL"
    return picked


# ---------- 解释生成 ----------

def build_explanation(per_skill: dict[str, dict], eff_weights: dict[str, float]) -> str:
    """按有效权重排序拼接各 Skill 理由（模板组装，零额外调用）。"""
    ordered = sorted(per_skill.items(), key=lambda kv: eff_weights.get(kv[0], 0), reverse=True)
    parts = [f"{sk}({row['score']:.1f}): {row['reason']}" for sk, row in ordered if row.get("reason")]
    return " | ".join(parts)


# ---------- 离线产物加载 ----------

def load_router_assets(cfg) -> dict:
    """加载敏感度矩阵与拟合权重（不存在则返回 None，调用方回退）。"""
    assets = {"sensitivity": None, "fitted_weights": None}
    sens_path = os.path.join(cfg.ladder_dir, "sensitivity.json")
    # 敏感度矩阵在 ladder eval 输出目录里；找最新一份
    evals = sorted(
        [d for d in os.listdir(cfg.ladder_dir) if d.startswith("eval_main")],
        reverse=True,
    ) if os.path.isdir(cfg.ladder_dir) else []
    for d in evals:
        p = os.path.join(cfg.ladder_dir, d, "sensitivity.json")
        if os.path.exists(p):
            with open(p) as f:
                assets["sensitivity"] = json.load(f)
            break
    w_path = os.path.join(cfg.runs_dir, "router_weights.json")
    if os.path.exists(w_path):
        with open(w_path) as f:
            assets["fitted_weights"] = json.load(f)
    return assets


# ---------- 共享工具函数（R6 脚本依赖） ----------

def read_scores_csv(path):
    """读取 scores.csv 返回 {img_id: row_dict}。"""
    import csv as _csv
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            rows[r["img_id"]] = r
    return rows


def opencv_features(img):
    """提取 7 维手工像素特征（无大模型）。"""
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    c = gray[1:-1, 1:-1]
    lap = -4 * c + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    lap_var = float(np.var(lap))
    box = (gray[:-2, :-2] + gray[:-2, 1:-1] + gray[:-2, 2:] + gray[1:-1, :-2] + c +
           gray[1:-1, 2:] + gray[2:, :-2] + gray[2:, 1:-1] + gray[2:, 2:]) / 9.0
    res = c - box
    gx = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2])
    thr = np.quantile(gx, 0.25)
    flat = res[gx <= thr]
    noise = float(1.4826 * np.median(np.abs(flat - np.median(flat)))) if flat.size else 0.0
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    h, w = gray.shape
    return {"lap_var": lap_var, "noise": noise, "colorful": colorful,
            "bright": float(gray.mean()), "logpix": float(np.log(h * w)), "aspect": float(w / h)}


def load_spaq_gate(cfg):
    """加载 SPAQ 软门控模型。文件不存在时返回等权回退。"""
    import numpy as np
    path = os.path.join(cfg.runs_dir, "router_v2", "route_spaq.json")
    if not os.path.exists(path):
        # 等权回退：3 槽位各 1/3（SPAQ 门控未过验证，等权是正确行为）
        return {
            "feat_keys": ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"],
            "mu": [0.0] * 7,
            "sd": [1.0] * 7,
            "W": [[0.0] * 7, [0.0] * 7, [0.0] * 7],
        }
    with open(path) as f:
        return json.load(f)


def gate_weights(g, feat):
    """计算 softmax 门控权重。g 来自 load_spaq_gate，feat 来自 opencv_features。"""
    import numpy as np
    raw = np.array([feat[k] for k in g["feat_keys"]], dtype=float)
    for j, k in enumerate(g["feat_keys"]):
        if k in ("lap_var", "noise", "colorful"):
            raw[j] = np.log(max(raw[j], 1e-6))
    z = (raw - np.array(g["mu"])) / np.array([s if s > 1e-6 else 1.0 for s in g["sd"]])
    zz = np.array(g["W"]) @ z
    zz = zz - zz.max()
    e = np.exp(zz)
    return e / e.sum()


def spaq_base_rows(cfg):
    """读取 SPAQ R1-bare / R1-rich / R2 评分缓存。"""
    r1b = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r1b_spaq", "scores.csv"))
    r1r = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r1r_spaq", "scores.csv"))
    r2 = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r2_spaq", "scores.csv"))
    return r1b, r1r, r2
