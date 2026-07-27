# -*- coding: utf-8 -*-
"""统一框架收官脚本。

步骤:
  1. 补 SPAQ BT 400 节点的 S-CONTENT 专家分 (~400 API, ¥1-2)
  2. 训练 SPAQ 3×7 门控矩阵 (本地, 0 API)
  3. 扫 SPAQ α (本地, 0 API)
  4. 跑统一 R6-SPAQ (全缓存, 0 API)
  6. 50_eval → 新主表 (第4次读MOS)
  7. 旧 R4/R5 从主表移除

用法: python scripts/98_unified_round.py
"""
import asyncio, csv, json, os, sys, time

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.prompts.skills import SKILLS, _PROCEDURE, _OUTPUT_CONTRACT, _SCALE_BLOCK
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics

POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]

BARE_PARAS = {
    "koniq": [
        f"Rate the overall quality of this image on a scale from 1 to 5. Reply with only a single number.",
        f"On a scale from 1 to 5, how would you rate the overall quality of this image? Reply with only a single number.",
        f"Give a single overall quality score for this image, from 1 (worst) to 5 (best). Reply with only the number.",
        f"As an image quality rater, assign one overall quality score from 1 to 5 to this image. Reply with only a single number.",
    ],
    "spaq": [
        f"Rate the overall quality of this image on a scale from 0 to 10. Reply with only a single number.",
        f"On a scale from 0 to 10, how would you rate the overall quality of this image? Reply with only a single number.",
        f"Give a single overall quality score for this image, from 0 (worst) to 10 (best). Reply with only the number.",
        f"As an image quality rater, assign one overall quality score from 0 to 10 to this image. Reply with only a single number.",
    ],
}


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


def opencv_features(img):
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


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def dynamic_fusion(W, skills3, feat_vec):
    """统一框架: 7特征 → 3权重 → 融合分。W=3×7矩阵, skills3=3专家分, feat_vec=7原始特征。"""
    f = np.array(feat_vec, dtype=float)
    g = softmax(W @ f)
    return float(g @ np.array(skills3)), g


# ═══════════════════════════════════════════════════════════════
# Gate training (identical to 93_train_router_v3 fit_gate)
# ═══════════════════════════════════════════════════════════════

def fit_gate(F, X, bt, train_idx, seed, steps=800, batch=256, lr=0.05, l2=1e-3):
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], F.shape[1]))
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
            coef = -1 / (1 + np.exp(-d))
            grad += coef * (np.outer(ga * (X[a] - sa), F[a]) - np.outer(gb * (X[b] - sb), F[b]))
            cnt += 1
        if cnt:
            W -= lr * (grad / cnt + 2 * l2 * W) / np.sqrt(t / 50 + 1)
    return W


def concordance(scores, bt, pairs):
    hits = sum(1 for i, j in pairs if (scores[i] - scores[j]) * (bt[i] - bt[j]) > 0)
    return hits / len(pairs)


# ═══════════════════════════════════════════════════════════════
# Step 1: SPAQ S-CONTENT 补分
# ═══════════════════════════════════════════════════════════════

async def step1_spaq_content(cfg):
    """补 SPAQ BT 400 节点缺少的 S-CONTENT 专家分。"""
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, "ranking_spaq.json"))
    node_scores = jload(os.path.join(wd, "spaq_node_scores.json"))

    cache_path = os.path.join(cfg.runs_dir, "posthoc", "spaq_bt_content.json")
    if os.path.exists(cache_path):
        cached = jload(cache_path)
    else:
        cached = {}

    # SPAQ BT 节点图片根目录
    img_dir = os.path.join(wd, "spaq_imgs")
    nodes = jload(os.path.join(wd, "nodes_spaq.json"))

    # Build image path map from nodes
    # nodes_spaq.json = [{"path": "spaq_imgs/xxxxx.jpg", ...}, ...]
    path_to_img = {}
    for n in nodes:
        # path is relative like "spaq_imgs/xxxxx.jpg"
        p = n["path"]
        # The actual file might be at the full path
        full_p = os.path.join(cfg.runs_dir, p) if not os.path.isabs(p) else p
        if os.path.exists(full_p):
            path_to_img[p] = full_p
        else:
            # Try alternative paths
            alt = os.path.join(wd, os.path.basename(p))
            if os.path.exists(alt):
                path_to_img[p] = alt
            # Also check spaq_train_images
            alt2 = os.path.join(cfg.runs_dir, "spaq_train_images", os.path.basename(p))
            if os.path.exists(alt2):
                path_to_img[p] = alt2

    # Build prompt
    def build_skill_prompt_offanchor(skill_id, scale_key):
        """简版: 维度+检查清单+等级+程序, 无_SCALE_BLOCK"""
        s = SKILLS[skill_id]
        parts = [
            f"You are an expert image quality assessor specialized as a {s['name']}.",
            f"[Your dimension]\n{s['dimension']}",
            "[Checklist — inspect each aspect]\n" + "\n".join(
                f"- {name}: {desc}" for name, desc in s["checklist"]
            ),
            "[Quality levels for YOUR dimension]\n" + "\n".join(
                f"  {i + 1} = {txt}" for i, txt in enumerate(reversed(s["levels"]), start=0)
            ),
            _PROCEDURE,
            _SCALE_BLOCK[scale_key],
            _OUTPUT_CONTRACT,
        ]
        return "\n\n".join(parts)

    prompt = build_skill_prompt_offanchor("S-CONTENT", "spaq")
    client = VLMClient(cfg, cfg.model_main)

    todo = []
    for r in ranking:
        p = r["path"]
        if p in cached:
            continue
        img_path = path_to_img.get(p)
        if img_path is None:
            continue
        todo.append((p, img_path))

    print(f"[step1] SPAQ S-CONTENT 待评: {len(todo)} 张")

    if todo:
        async def one(p, img_path):
            text, _ = await client.score_image(img_path, prompt, temperature=0.0)
            parsed = parse_score(text, (0, 10))
            return p, (parsed["score"] if parsed else None)

        jobs = [one(p, ip) for p, ip in todo]
        for c0 in range(0, len(jobs), 200):
            part = await gather_with_progress(jobs[c0:c0 + 200], every=50, label=f"spaq-content[{c0//200}]")
            for r in part:
                if not isinstance(r, Exception) and r[1] is not None:
                    cached[r[0]] = r[1]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached, f)

    print(f"[step1] S-CONTENT 分: {len(cached)}/{len(ranking)}")
    return cached, path_to_img, client


# ═══════════════════════════════════════════════════════════════
# Step 2: SPAQ 3×7 gate training
# ═══════════════════════════════════════════════════════════════

def step2_train_spaq_gate(cfg, content_cache):
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, "ranking_spaq.json"))
    node_scores = jload(os.path.join(wd, "spaq_node_scores.json"))
    nodes = jload(os.path.join(wd, "nodes_spaq.json"))
    route_feats = jload(os.path.join(wd, "route_features_spaq.json")) if os.path.exists(
        os.path.join(wd, "route_features_spaq.json")) else {}

    # Build path→feat map
    node_paths = {n["path"]: n for n in nodes}

    # Assemble training data
    data = []
    n_missing_feat = 0
    for r in ranking:
        p = r["path"]
        ss = node_scores.get(p)
        s_content = content_cache.get(p)
        if not ss or s_content is None:
            continue
        skills = {**ss, "S-CONTENT": s_content}
        if any(skills.get(sk) is None for sk in POOL):
            continue

        # Get features
        feats = route_feats.get(p)
        if not feats or "noise" not in feats:
            # Compute from image
            img_path = None
            for candidate in [
                os.path.join(wd, os.path.basename(p)),
                os.path.join(cfg.runs_dir, "spaq_train_images", os.path.basename(p)),
                os.path.join(cfg.runs_dir, "full_tournament", "spaq_imgs", os.path.basename(p)),
            ]:
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                n_missing_feat += 1
                continue
            try:
                feats = opencv_features(Image.open(img_path))
            except Exception:
                n_missing_feat += 1
                continue
        data.append({"path": p, "bt": r["bt"], "skills": skills, "feat": feats})

    n = len(data)
    print(f"[step2] SPAQ BT有效节点: {n} (缺特征: {n_missing_feat})")

    if n < 50:
        print("[step2] 样本不足, 门控 FAIL → 回退等权")
        return None, n

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

    # Split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    train_idx = perm[:n // 2]
    valid_idx = perm[n // 2:]
    vset = set(valid_idx.tolist())

    # Train
    W = fit_gate(F, X, bt, train_idx, 42)

    # Validate
    rng_p = np.random.default_rng(43)
    pairs = []
    while len(pairs) < min(2000, n * 3):
        i, j = rng_p.integers(0, n, 2)
        if i in vset and j in vset and abs(bt[i] - bt[j]) >= np.std(bt) * 0.1:
            pairs.append((i, j))

    scores_dyn = np.array([softmax(W @ F[i]) @ X[i] for i in range(n)])
    scores_eq = X.mean(axis=1)
    c_dyn = concordance(scores_dyn, bt, pairs)
    c_eq = concordance(scores_eq, bt, pairs)

    print(f"[step2] 留出一致率: 动态={c_dyn:.4f}  等权={c_eq:.4f}")

    if c_dyn <= c_eq + 0.005:
        print(f"[step2] 门控 FAIL (动态 ≤ 等权+0.005) → 等权回退")
        return None, n

    gate_model = {
        "W": W.tolist(),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "feat_keys": FEAT_KEYS,
        "pool": POOL,
    }
    out_path = os.path.join(cfg.runs_dir, "router_v3", "fusion_spaq.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gate_model, f, ensure_ascii=False, indent=2)
    print(f"[step2] SPAQ 门控 PASS → {out_path}")
    return gate_model, n


# ═══════════════════════════════════════════════════════════════
# Step 3: Scan α for SPAQ
# ═══════════════════════════════════════════════════════════════

def step3_scan_alpha(cfg, gate_model, ds):
    """在 BT 训练集的留出集上扫最优 α。gate_model=None → 等权。"""
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    rank_file = "ranking_koniq_v3.json" if ds == "koniq" else "ranking_spaq.json"
    ranking = jload(os.path.join(wd, rank_file))
    ws = jload(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"))
    new_sc = jload(os.path.join(wd, "new_node_scores.json"))
    feats = jload(os.path.join(wd, "route_features_koniq.json"))
    feats_v3 = jload(os.path.join(wd, "features_koniq_v3.json"))
    ws_by_path = {r["path"]: r for r in ws}

    data = []
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
        if not isinstance(ft, dict) or "noise" not in ft:
            continue
        data.append({"path": p, "bt": r["bt"], "skills": {sk: ss[sk] for sk in POOL}, "feat": ft})

    n = len(data)
    # Build feature matrix
    spread_arr = np.array([np.std([d["skills"][sk] for sk in POOL]) for d in data])
    raw = np.array([[d["feat"]["lap_var"], d["feat"]["noise"], d["feat"]["colorful"],
                     d["feat"]["bright"], d["feat"]["logpix"], d["feat"]["aspect"], 0.0] for d in data])
    raw[:, 6] = spread_arr
    for j, k in enumerate(FEAT_KEYS):
        if k in ("lap_var", "noise", "colorful"):
            raw[:, j] = np.log(np.maximum(raw[:, j], 1e-6))
    mu = raw.mean(axis=0); sd = raw.std(axis=0) + 1e-9
    F = (raw - mu) / sd
    X = np.array([[d["skills"][sk] for sk in POOL] for d in data])
    bt = np.array([d["bt"] for d in data])

    W = np.array(gate_model["W"]) if gate_model else None

    # Compute dynamic fusion scores
    if W is not None:
        scores_dyn = np.array([softmax(W @ F[i]) @ X[i] for i in range(n)])
    else:
        scores_dyn = X.mean(axis=1)  # 等权

    # bare vote isn't available for BT nodes (no paras). Use R1-bare-like single score.
    # For α scan, we use the S-GLOBAL expert score as a proxy for "bare-like" scoring.
    # In practice during eval, bare vote is from cached paras. Here we approximate.
    scores_bare = X[:, POOL.index("S-GLOBAL")]  # proxy

    # Split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    valid_idx = perm[n // 2:]
    vset = set(valid_idx.tolist())

    rng_p = np.random.default_rng(44)
    pairs = []
    while len(pairs) < 2000:
        i, j = rng_p.integers(0, n, 2)
        if i in vset and j in vset and abs(bt[i] - bt[j]) >= np.std(bt) * 0.1:
            pairs.append((i, j))

    best_alpha, best_hits = 0.6, 0
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        scores = alpha * scores_dyn + (1 - alpha) * scores_bare
        hits = concordance(scores, bt, pairs)
        if hits > best_hits:
            best_hits = hits
            best_alpha = alpha

    print(f"[step3] {ds} 最优α = {best_alpha:.2f} (一致率={best_hits:.4f})")
    return best_alpha


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

def run_unified_arm(cfg, ds, eval_ds, gate_model, alpha, offanchor=False):
    """给一个域跑统一框架: 3专家 dynamic_fusion + bare投票。offanchor=True → 砍_SCALE_BLOCK。"""
    lo, hi = cfg.scales[ds]
    images = {r.img_id: r.path for r in load_images(cfg, eval_ds)}

    # Bare voting
    r1b_path = os.path.join(cfg.runs_dir, "final", f"r1b_{ds}", "scores.csv")
    with open(r1b_path, encoding="utf-8-sig") as f:
        r1b = {r["img_id"]: float(r["score"]) for r in csv.DictReader(f) if r.get("score") not in ("", None)}

    paras_path = os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json")
    paras = jload(paras_path) if os.path.exists(paras_path) else {}

    # Expert scores - from R2 cache (original anchored) or offanchor cache
    if offanchor:
        expert_cache = jload(expert_cache_path) if os.path.exists(expert_cache_path) else {}
    else:
        # Read from R2 cache (original anchored experts)
        r2_path = os.path.join(cfg.runs_dir, "final", f"r2_{ds}", "scores.csv")
        expert_cache = {}
        with open(r2_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("skill_scores"):
                    ss = json.loads(r["skill_scores"])
                    expert_cache[r["img_id"]] = {sk: ss[sk] for sk in POOL if sk in ss}

    if gate_model is not None:
        W = np.array(gate_model["W"])
        mu_arr = np.array(gate_model["mu"])
        sd_arr = np.array([s if s > 1e-6 else 1.0 for s in gate_model["sd"]])
    else:
        W = None

    ids = sorted(set(expert_cache.keys()) & set(r1b.keys()) & set(images.keys()))
    rows = []
    for img_id in ids:
        # bare vote
        bare_parts = [r1b[img_id]] + [paras.get(img_id, {}).get(str(k)) for k in (1, 2, 3)]
        if any(v is None for v in bare_parts):
            continue
        vote = float(np.mean(bare_parts))

        # expert scores
        es = expert_cache[img_id]
        skills3 = [es[sk] for sk in POOL]
        if any(v is None for v in skills3):
            continue

        feat_raw = opencv_features(Image.open(images[img_id]))
        spread_val = float(np.std(skills3))
        f = np.array([feat_raw["lap_var"], feat_raw["noise"], feat_raw["colorful"],
                      feat_raw["bright"], feat_raw["logpix"], feat_raw["aspect"], spread_val], dtype=float)
        for j, k in enumerate(FEAT_KEYS):
            if k in ("lap_var", "noise", "colorful"):
                f[j] = np.log(max(f[j], 1e-6))

        if W is not None:
            f_std = (f - mu_arr) / sd_arr
            fus, g = dynamic_fusion(W, skills3, f_std)
        else:
            fus = float(np.mean(skills3))
            g = np.ones(3) / 3

        final = alpha * fus + (1 - alpha) * vote
        tag = "offanchor" if offanchor else "unified"
        reason = (f"R6-{tag}=α={alpha:.1f}·动态融合("
                  f"{','.join(f'{s}×{w:.2f}' for s, w in zip(POOL, g))})"
                  f"+{1-alpha:.1f}·bare投票({vote:.2f})")
        rows.append({"img_id": img_id, "score": round(final, 4), "reason": reason})

    return rows


def write_arm_output(cfg, arm_name, ds, rows):
    out_dir = os.path.join(cfg.runs_dir, "posthoc", arm_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score", "reason", "parse_tier"])
        w.writeheader()
        for r in rows:
            w.writerow({"img_id": r["img_id"], "dataset": ds, "route": arm_name,
                        "score": r["score"], "reason": r.get("reason", "")[:300], "parse_tier": 1})
    return out_dir


def eval_arm(cfg, arm_dir, ds, eval_ds):
    with open(os.path.join(arm_dir, "scores.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pred = {r["img_id"]: float(r["score"]) for r in rows}
    mos = load_mos(cfg, eval_ds)
    m = compute_metrics(pred, mos)
    vals = np.array(list(pred.values()))
    summary = {
        "arm": os.path.basename(arm_dir), "n": m["n"],
        "SRCC": round(m["SRCC"], 4), "MAE": round(m["MAE"], 4), "PLCC": round(m["PLCC"], 4),
        "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
        "unique": int(len(set(np.round(vals, 4)))),
    }
    with open(os.path.join(arm_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main_async():
    cfg = get_config()
    out_root = os.path.join(cfg.runs_dir, "posthoc")
    os.makedirs(out_root, exist_ok=True)

    # ── Step 1: SPAQ S-CONTENT ──
    content_cache, spaq_img_map, client = await step1_spaq_content(cfg)

    # ── Step 2: SPAQ gate ──
    gate_spaq, n_spaq = step2_train_spaq_gate(cfg, content_cache)

    # ── Step 3: Scan α for both domains ──
    gate_koniq = jload(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"))
    alpha_k = step3_scan_alpha(cfg, gate_koniq, "koniq")
    alpha_s = step3_scan_alpha(cfg, gate_spaq, "spaq") if gate_spaq else 0.6

    if gate_spaq is None:
        print("[main] SPAQ 门控 FAIL → 等权回退, α=0.6")
        alpha_s = 0.6

    # ── Step 4: Unified R6 both domains ──
    print("\n=== 统一 R6 ===")
    rows_k = run_unified_arm(cfg, "koniq", "koniq_val", gate_koniq, alpha_k, offanchor=False)
    dir_k = write_arm_output(cfg, "r6_unified_koniq", "koniq", rows_k)
    s_k = eval_arm(cfg, dir_k, "koniq", "koniq_val")
    print(f"  KonIQ: SRCC={s_k['SRCC']} MAE={s_k['MAE']} PLCC={s_k['PLCC']} n={s_k['n']}")

    rows_s = run_unified_arm(cfg, "spaq", "spaq_test", gate_spaq, alpha_s, offanchor=False)
    dir_s = write_arm_output(cfg, "r6_unified_spaq", "spaq", rows_s)
    s_s = eval_arm(cfg, dir_s, "spaq", "spaq_test")
    print(f"  SPAQ:  SRCC={s_s['SRCC']} MAE={s_s['MAE']} PLCC={s_s['PLCC']} n={s_s['n']}")

    rows_ok = run_unified_arm(cfg, "koniq", "koniq_val", gate_koniq, alpha_k, offanchor=True)
    s_ok = eval_arm(cfg, dir_ok, "koniq", "koniq_val")
    print(f"  KonIQ: SRCC={s_ok['SRCC']} MAE={s_ok['MAE']} PLCC={s_ok['PLCC']} n={s_ok['n']}")

    rows_os = run_unified_arm(cfg, "spaq", "spaq_test", gate_spaq, alpha_s, offanchor=True)
    s_os = eval_arm(cfg, dir_os, "spaq", "spaq_test")
    print(f"  SPAQ:  SRCC={s_os['SRCC']} MAE={s_os['MAE']} PLCC={s_os['PLCC']} n={s_os['n']}")

    # ── Step 6: 总表 ──
    print("\n" + "=" * 70)
    print("统一框架主表")
    print("=" * 70)
    print(f"{'臂':<30} {'KonIQ SRCC':>10} {'KonIQ MAE':>10} {'KonIQ PLCC':>10} {'SPAQ SRCC':>10} {'SPAQ MAE':>10} {'SPAQ PLCC':>10}")
    print("-" * 70)
    # Read old main table for R1-R3
    old = {}
    with open(os.path.join(cfg.runs_dir, "final", "main_table.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["run"], r["dataset"])
            old[key] = r

    for arm_label, koniq_key, spaq_key in [
        ("R1-bare", "r1b_koniq", "r1b_spaq"),
        ("R1-rich", "r1r_koniq", "r1r_spaq"),
        ("R2", "r2_koniq", "r2_spaq"),
        ("R2.5", "r25_koniq", "r25_spaq"),
        ("R3", "r3_koniq", "r3_spaq"),
    ]:
        rk = old.get((koniq_key, "koniq_val"), {})
        rs = old.get((spaq_key, "spaq_test"), {})
        print(f"{arm_label:<30} {rk.get('SRCC','?'):>10} {rk.get('MAE','?'):>10} {rk.get('PLCC','?'):>10} {rs.get('SRCC','?'):>10} {rs.get('MAE','?'):>10} {rs.get('PLCC','?'):>10}")

    print(f"{'R6 (统一框架)':<30} {s_k['SRCC']:>10.4f} {s_k['MAE']:>10.4f} {s_k['PLCC']:>10.4f} {s_s['SRCC']:>10.4f} {s_s['MAE']:>10.4f} {s_s['PLCC']:>10.4f}")

    print(f"\nKonIQ: α={alpha_k:.1f}  SPAQ: α={alpha_s:.1f}  gate={'PASS' if gate_spaq else 'FAIL→等权'}")
    print(f"账本: {client.ledger() if 'client' in dir() else 'N/A'}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
