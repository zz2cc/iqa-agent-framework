# -*- coding: utf-8 -*-
"""二轮 S4：Router v2 训练（本地，零 API）。

合规（ADR-0003 §5）：训练仅发生在 Router/Decision 层；监督信号全部纯像素自衍生
（BT 排行榜 + 梯子 PASS 族已知排序），零 MOS。

训练对象（docs/二轮重设计计划-通俗版.md §4）：
- 任务 A 冲突裁决：5 专家融合权重 w（R^5, ≥0），目标=融合分与 BT 排行榜的成对一致率
  最大（hinge loss，节点级 split-half 验证；对照=等权/一轮 trimmed mean）。
- 任务 B 规则选择：协议路由（bare / rich / multi）——需要各协议在工作集上的分数
  （先由 87_score_protocols.py 补齐，~¥7）。按验证集一致率决定每图协议或全局门控。
- 留一信号归因（v1.1 §10.1）：梯子-only / BT-only / 梯子+BT 三组训练对照。

输入产物：runs/full_tournament/ranking_{dom}.json（BT 排行榜）、
  runs/bt_pilot/workset_scores.json（KonIQ 5 专家分）、
  runs/full_tournament/nodes_spaq.json（SPAQ 2 技能分）、
  runs/ladder2/scores.json + endpoint_accuracy.json（梯子 PASS 族）。
输出：runs/router_v2/weights.json + train_report.json。

用法：
  python scripts/88_train_router_v2.py --task fusion --domain koniq
  python scripts/88_train_router_v2.py --task fusion --domain koniq --loso
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config

SKILLS = ["S-TECH", "S-AESTH", "S-CONTENT", "S-NATURAL", "S-GLOBAL"]


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


def load_domain_rows(cfg, dom):
    """返回 [{skills: {sk: score}, bt: float}]（KonIQ 5 技能；SPAQ 2 技能）。"""
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, f"ranking_{dom}.json"))
    if dom == "koniq":
        ws = jload(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"))
        by_path = {r["path"]: r for r in ws}
        rows = []
        for r in ranking:
            src = by_path.get(r["path"])
            if not src:
                continue
            ss = json.loads(src["skill_scores"]) if isinstance(src["skill_scores"], str) else src["skill_scores"]
            rows.append({"skills": ss, "bt": r["bt"], "old": r["old"]})
        return rows
    sc_map = jload(os.path.join(wd, "spaq_node_scores.json"))
    rows = []
    for r in ranking:
        ss = sc_map.get(r["path"], {})
        if ss:
            rows.append({"skills": ss, "bt": r["bt"], "old": r["old"]})
    return rows


def score_matrix(rows):
    X = np.array([[r["skills"].get(sk, np.nan) for sk in SKILLS] for r in rows])
    return X


def pairwise_concordance(scores, bt, pairs):
    hits = 0
    for i, j in pairs:
        if (scores[i] - scores[j]) * (bt[i] - bt[j]) > 0:
            hits += 1
    return hits / len(pairs)


def sample_pairs(bt, n_pairs, seed, min_margin=0.5):
    rng = np.random.default_rng(seed)
    n = len(bt)
    pairs = []
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, n, 2)
        if abs(bt[i] - bt[j]) >= min_margin:
            pairs.append((i, j) if bt[i] > bt[j] else (j, i))
    return pairs


def fit_fusion_weights(X, bt, train_idx, valid_pairs, seed, iters=400, lr=0.05, l2=1e-3):
    """pairwise hinge：对每对 (i,j)（BT 说 i>j），要求 w·x_i - w·x_j > margin。"""
    rng = np.random.default_rng(seed)
    w = np.ones(X.shape[1]) / X.shape[1]
    mask = ~np.isnan(X).any(axis=1)
    tr = [i for i in train_idx if mask[i]]
    for t in range(iters):
        i, j = rng.choice(tr, 2, replace=False)
        if bt[i] == bt[j]:
            continue
        a, b = (i, j) if bt[i] > bt[j] else (j, i)
        margin = 1.0 - float(w @ (X[a] - X[b]))
        if margin > 0:
            grad = -(X[a] - X[b]) + 2 * l2 * w
            w = np.maximum(0.0, w - lr * grad / np.sqrt(t + 1))
    s = w.sum()
    return w / s if s > 0 else np.ones(X.shape[1]) / X.shape[1]


def cmd_fusion(cfg, dom, loso):
    rows = load_domain_rows(cfg, dom)
    # 动态确定可用技能（SPAQ 只有 S-TECH/S-GLOBAL 两技能）
    active = [sk for sk in SKILLS
              if np.mean([r["skills"].get(sk) is not None for r in rows]) > 0.9]
    X = np.array([[r["skills"][sk] for sk in active] for r in rows])
    bt = np.array([r["bt"] for r in rows])
    n = len(rows)
    print(f"[fusion:{dom}] 样本 {n}，技能 {active}")
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    train_idx, valid_idx = perm[: n // 2], perm[n // 2:]
    valid_pairs = sample_pairs(bt, 20000, cfg.seed + 1)[::2]
    valid_pairs = [(i, j) for i, j in valid_pairs if i in set(valid_idx) and j in set(valid_idx)][:5000]

    eq = X.mean(axis=1)
    old = np.array([r["old"] for r in rows])
    c_eq = pairwise_concordance(eq, bt, valid_pairs)
    c_old = pairwise_concordance(old, bt, valid_pairs)

    w = fit_fusion_weights(X, bt, train_idx, valid_pairs, cfg.seed)
    fused = X @ w
    c_w = pairwise_concordance(fused, bt, valid_pairs)

    report = {
        "domain": dom, "n": n, "active_skills": active, "n_valid_pairs": len(valid_pairs),
        "concordance_equal_weights": round(float(c_eq), 4),
        "concordance_old_fusion": round(float(c_old), 4),
        "concordance_learned": round(float(c_w), 4),
        "weights": {sk: round(float(wi), 4) for sk, wi in zip(active, w)},
        "srcc_learned_vs_bt": round(float(spearmanr(fused, bt).statistic), 4),
        "gate_pass": bool(c_w > c_eq + 0.01),
    }
    out = os.path.join(cfg.runs_dir, "router_v2")
    os.makedirs(out, exist_ok=True)
    json.dump({"weights": report["weights"]}, open(os.path.join(out, f"weights_{dom}.json"), "w"))
    json.dump(report, open(os.path.join(out, f"fusion_report_{dom}.json"), "w"), indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[fusion:{dom}] 门控：学习权重须比等权高 +0.01 → {'PASS' if report['gate_pass'] else 'FAIL（回退等权）'}")


# ---------- 任务 B：协议路由（软门控，§4.3 评估规则的选择） ----------

PROTOCOLS = ["bare", "rich", "multi"]


def opencv_features(path: str) -> dict:
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


def route_scores(W, F, P):
    n = F.shape[0]
    out = np.zeros(n)
    gates = np.zeros((n, P.shape[1]))
    for i in range(n):
        g = softmax(W @ F[i])
        gates[i] = g
        out[i] = float(g @ P[i])
    return out, gates


def fit_route(W0, F, P, bt, train_idx, seed, steps=600, batch=256, lr=0.05, l2=1e-3):
    """pairwise hinge：BT 说 a>b 时要求 route(a)-route(b)>0。解析梯度（softmax 雅可比）。"""
    rng = np.random.default_rng(seed)
    W = W0.copy()

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
            sa, sb = float(ga @ P[a]), float(gb @ P[b])
            d = sa - sb
            coef = -sigmoid(-d)
            grad += coef * (np.outer(ga * (P[a] - sa), F[a]) - np.outer(gb * (P[b] - sb), F[b]))
            cnt += 1
        if cnt:
            W -= lr * (grad / cnt + 2 * l2 * W) / np.sqrt(t / 50 + 1)
    return W


def load_domain_rows_with_paths(cfg, dom):
    """route 专用：带 path 的全字段行。"""
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, f"ranking_{dom}.json"))
    if dom == "koniq":
        ws = jload(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"))
        by_path = {r["path"]: r for r in ws}
        rows = []
        for r in ranking:
            src = by_path.get(r["path"])
            if not src:
                continue
            ss = json.loads(src["skill_scores"]) if isinstance(src["skill_scores"], str) else src["skill_scores"]
            rows.append({"path": r["path"], "skills": ss, "bt": r["bt"]})
        return rows
    sc_map = jload(os.path.join(wd, "spaq_node_scores.json"))
    return [{"path": r["path"], "skills": sc_map.get(r["path"], {}), "bt": r["bt"]}
            for r in ranking if sc_map.get(r["path"])]


def cmd_route(cfg, dom):
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    proto = jload(os.path.join(wd, f"protocol_scores_{dom}.json"))
    wpath = os.path.join(cfg.runs_dir, "router_v2", f"weights_{dom}.json")
    weights = json.load(open(wpath))["weights"] if os.path.exists(wpath) else {}

    fpath = os.path.join(wd, f"route_features_{dom}.json")
    feats = jload(fpath) if os.path.exists(fpath) else {}
    data = []
    for r in load_domain_rows_with_paths(cfg, dom):
        ps = proto.get(r["path"])
        if not ps or ps.get("bare") is None or ps.get("rich") is None:
            continue
        if r["path"] not in feats:
            feats[r["path"]] = opencv_features(r["path"])
        multi = float(sum(r["skills"][sk] * weights.get(sk, 0.0) for sk in r["skills"])) \
            if weights else float(np.mean(list(r["skills"].values())))
        data.append({"path": r["path"], "bt": r["bt"], "P": [ps["bare"], ps["rich"], multi]})
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(feats, f)
    n = len(data)
    print(f"[route:{dom}] 样本 {n}")

    FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect"]
    raw = np.array([[feats[d["path"]][k] for k in FEAT_KEYS] for d in data])
    for j, k in enumerate(FEAT_KEYS):
        if k in ("lap_var", "noise", "colorful"):
            raw[:, j] = np.log(np.maximum(raw[:, j], 1e-6))
    mu, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-9
    Fz = (raw - mu) / sd
    P = np.array([d["P"] for d in data])
    bt = np.array([d["bt"] for d in data])

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    train_idx, valid_idx = perm[: n // 2], perm[n // 2:]
    valid_pairs = sample_pairs(bt, 20000, cfg.seed + 2)[::2]
    vset = set(valid_idx.tolist())
    valid_pairs = [(i, j) for i, j in valid_pairs if i in vset and j in vset][:5000]

    conc = {name: pairwise_concordance(P[:, k], bt, valid_pairs) for k, name in enumerate(PROTOCOLS)}
    best_single = max(conc, key=conc.get)

    W = fit_route(np.zeros((len(PROTOCOLS), Fz.shape[1])), Fz, P, bt, train_idx, cfg.seed)
    routed, gates = route_scores(W, Fz, P)
    c_routed = pairwise_concordance(routed, bt, valid_pairs)

    W2 = fit_route(np.zeros_like(W), Fz, P, bt, valid_idx, cfg.seed + 1)
    routed2, _ = route_scores(W2, Fz, P)
    stab = float(spearmanr(routed[valid_idx], routed2[valid_idx]).statistic)

    report = {
        "domain": dom, "n": n, "n_valid_pairs": len(valid_pairs),
        "concordance_single": {k: round(float(v), 4) for k, v in conc.items()},
        "best_single": best_single,
        "concordance_routed": round(float(c_routed), 4),
        "gate_pass": bool(c_routed > conc[best_single] + 0.01),
        "route_stability_srcc": round(stab, 4),
        "gate_mean": {name: round(float(gates[:, k].mean()), 3) for k, name in enumerate(PROTOCOLS)},
        "W": [[round(float(x), 4) for x in row] for row in W],
        "feat_keys": FEAT_KEYS, "feat_mu": [round(float(x), 4) for x in mu],
        "feat_sd": [round(float(x), 4) for x in sd],
    }
    out = os.path.join(cfg.runs_dir, "router_v2")
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, f"route_report_{dom}.json"), "w"), indent=1)
    json.dump({"W": report["W"], "feat_keys": FEAT_KEYS, "mu": report["feat_mu"],
               "sd": report["feat_sd"], "protocols": PROTOCOLS, "best_single": best_single,
               "fallback": None if report["gate_pass"] else best_single},
              open(os.path.join(out, f"route_{dom}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "W"}, ensure_ascii=False, indent=1))
    print(f"[route:{dom}] 门控：软路由须比最强单协议 +0.01 → {'PASS' if report['gate_pass'] else f'FAIL（回退 {best_single}）'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fusion", choices=["fusion", "route"])
    ap.add_argument("--domain", required=True, choices=["koniq", "spaq"])
    ap.add_argument("--loso", action="store_true", help="留一信号归因（后续版本）")
    args = ap.parse_args()
    cfg = get_config()
    if args.task == "fusion":
        cmd_fusion(cfg, args.domain, args.loso)
    elif args.task == "route":
        cmd_route(cfg, args.domain)


if __name__ == "__main__":
    main()
