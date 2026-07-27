# -*- coding: utf-8 -*-
"""V3-D2：Router v3 逐图条件化融合（§4.3 冲突裁决，本地零 API）。

设计（V3-B）：从"全场一套静态权重"升级为"每张图一套权重"——
软门控：6 维 OpenCV 特征 + 专家分歧统计 → 每图 3 专家（TECH/GLOBAL/CONTENT，V3-C 瘦身池）
的凸组合权重，BT 排行榜监督（pairwise hinge）。对照：等权 / 静态 3 池权重 / v2 五池权重。
门控：动态 > 静态 +0.01，否则回退静态；两半稳定性另报。

输入：runs/full_tournament/ranking_koniq_v3.json、new_node_scores.json、
  bt_pilot/workset_scores.json、route_features_koniq.json、features_koniq_v3.json。
输出：runs/router_v3/{weights,report}.json。

用法： python scripts/93_train_router_v3.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config

POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


def full_cv_features(path):
    """与 88 同实现的全量 OpenCV 特征（92 只存了 lap_var，此处补全）。"""
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
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


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_static(X, bt, train_idx, seed, iters=600, lr=0.05, l2=1e-3):
    rng = np.random.default_rng(seed)
    w = np.ones(X.shape[1]) / X.shape[1]
    for t in range(iters):
        i, j = rng.choice(train_idx, 2, replace=False)
        if bt[i] == bt[j]:
            continue
        a, b = (i, j) if bt[i] > bt[j] else (j, i)
        margin = 1.0 - float(w @ (X[a] - X[b]))
        if margin > 0:
            w = np.maximum(0.0, w - lr * (-(X[a] - X[b]) + 2 * l2 * w) / np.sqrt(t + 1))
    s = w.sum()
    return w / s if s > 0 else np.ones(X.shape[1]) / X.shape[1]


def fit_gate(F, X, bt, train_idx, seed, steps=800, batch=256, lr=0.05, l2=1e-3):
    """逐图门控：g_i = softmax(W f_i)，final_i = g_i·x_i。pairwise hinge + 解析梯度。"""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], F.shape[1]))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for t in range(steps):
        idx = rng.choice(train_idx, min(batch, len(train_idx)), replace=False)
        grad = np.zeros_like(W)
        cnt = 0
        for k in range(0, len(idx) - 1, 2):
            i, j = idx[k], idx[k + 1]
            if bt[i] == bt[j]:
                continue
            a, b = (i, j) if bt[i] > bt[j] else (j, i)
            ga, gb = softmax(W @ F[a]), softmax(W @ F[b])
            sa, sb = float(ga @ X[a]), float(gb @ X[b])
            d = sa - sb
            coef = -sigmoid(-d)
            grad += coef * (np.outer(ga * (X[a] - sa), F[a]) - np.outer(gb * (X[b] - sb), F[b]))
            cnt += 1
        if cnt:
            W -= lr * (grad / cnt + 2 * l2 * W) / np.sqrt(t / 50 + 1)
    return W


def concordance(scores, bt, pairs):
    hits = sum(1 for i, j in pairs if (scores[i] - scores[j]) * (bt[i] - bt[j]) > 0)
    return hits / len(pairs)


def main():
    cfg = get_config()
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, "ranking_koniq_v3.json"))
    ws = jload(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"))
    new_sc = jload(os.path.join(wd, "new_node_scores.json"))
    feats = jload(os.path.join(wd, "route_features_koniq.json"))
    feats_v3 = jload(os.path.join(wd, "features_koniq_v3.json"))
    ws_by_path = {r["path"]: r for r in ws}

    # 装配：3 专家分 + BT + 特征（老节点取 5 技能中的 3；新节点原生 3 技能）
    data = []
    n_feat_backfill = 0
    for r in ranking:
        p = r["path"]
        if p in ws_by_path:
            ss = json.loads(ws_by_path[p]["skill_scores"]) if isinstance(ws_by_path[p]["skill_scores"], str) else ws_by_path[p]["skill_scores"]
            ft = feats.get(p)
        else:
            ss = new_sc.get(p)
            ft = feats_v3.get(str(r["node"]))
        if not ss or any(ss.get(sk) is None for sk in POOL):
            continue
        if not isinstance(ft, dict) or "noise" not in ft:  # 92 只存了 lap_var 裸值 → 本地补全量
            ft = full_cv_features(p)
            n_feat_backfill += 1
        data.append({"path": p, "bt": r["bt"], "skills": {sk: ss[sk] for sk in POOL}, "feat": ft})
    n = len(data)
    print(f"[router_v3] 节点 {n}（特征补全 {n_feat_backfill} 张）")

    X = np.array([[d["skills"][sk] for sk in POOL] for d in data])
    bt = np.array([d["bt"] for d in data])
    spread = X.std(axis=1)
    raw = np.array([[d["feat"]["lap_var"], d["feat"]["noise"], d["feat"]["colorful"],
                     d["feat"]["bright"], d["feat"]["logpix"], d["feat"]["aspect"], 0.0] for d in data])
    raw[:, 6] = spread
    for j, k in enumerate(FEAT_KEYS):
        if k in ("lap_var", "noise", "colorful"):
            raw[:, j] = np.log(np.maximum(raw[:, j], 1e-6))
    mu, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-9
    F = (raw - mu) / sd

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    train_idx, valid_idx = perm[: n // 2], perm[n // 2:]
    vset = set(valid_idx.tolist())
    rng_p = np.random.default_rng(cfg.seed + 1)
    pairs = []
    while len(pairs) < 8000:
        i, j = rng_p.integers(0, n, 2)
        if i in vset and j in vset and abs(bt[i] - bt[j]) >= 0.5:
            pairs.append((i, j))

    eq = X.mean(axis=1)
    w_static = fit_static(X, bt, train_idx, cfg.seed)
    st = X @ w_static
    W = fit_gate(F, X, bt, train_idx, cfg.seed)
    dyn = np.array([softmax(W @ F[i]) @ X[i] for i in range(n)])
    W2 = fit_gate(F, X, bt, valid_idx, cfg.seed + 1)
    dyn2 = np.array([softmax(W2 @ F[i]) @ X[i] for i in range(n)])

    c_eq, c_st, c_dyn = concordance(eq, bt, pairs), concordance(st, bt, pairs), concordance(dyn, bt, pairs)
    stab = float(spearmanr(dyn[valid_idx], dyn2[valid_idx]).statistic)
    report = {
        "n": n, "pool": POOL,
        "concordance_equal": round(float(c_eq), 4),
        "concordance_static_3pool": round(float(c_st), 4),
        "concordance_dynamic": round(float(c_dyn), 4),
        "static_weights": {sk: round(float(w), 4) for sk, w in zip(POOL, w_static)},
        "gate_pass": bool(c_dyn > c_st + 0.01),
        "stability_srcc": round(stab, 4),
        "gate_mean": {sk: round(float(np.mean([softmax(W @ F[i])[k] for i in range(n)])), 3) for k, sk in enumerate(POOL)},
        "feat_keys": FEAT_KEYS, "mu": [round(float(x), 4) for x in mu], "sd": [round(float(x), 4) for x in sd],
        "W": [[round(float(x), 4) for x in row] for row in W],
    }
    out = os.path.join(cfg.runs_dir, "router_v3")
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "report_koniq.json"), "w"), indent=1)
    json.dump({"W": report["W"], "static_weights": report["static_weights"], "feat_keys": FEAT_KEYS,
               "mu": report["mu"], "sd": report["sd"], "pool": POOL,
               "fallback": None if report["gate_pass"] else "static"},
              open(os.path.join(out, "fusion_koniq.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "W"}, ensure_ascii=False, indent=1))
    print(f"[router_v3] 门控：动态须 > 静态 +0.01 → {'PASS（逐图动态上场）' if report['gate_pass'] else 'FAIL（回退静态 3 池）'}")


if __name__ == "__main__":
    main()
