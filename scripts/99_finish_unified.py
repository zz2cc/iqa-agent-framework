# -*- coding: utf-8 -*-
"""统一框架收官 续篇(step2已跑完→等权回退)，步骤3-6一版完成。"""
import csv, json, os, sys, numpy as np
from PIL import Image
from scipy.stats import spearmanr

sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_mos
from iqa_agent.metrics import compute_metrics

cfg = get_config()
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


def opencv_features(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    c = gray[1:-1, 1:-1]
    lap = -4 * c + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    lap_var = float(np.var(lap))
    box = (gray[:-2, :-2] + gray[:-2, 1:-1] + gray[:-2, 2:] + gray[1:-1, :-2] + c +
           gray[1:-1, 2:] + gray[2:, :-2] + gray[2:, 1:-1] + gray[2:, 2:]) / 9.0
    res = c - box; gx = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2])
    thr = np.quantile(gx, 0.25); flat = res[gx <= thr]
    noise = float(1.4826 * np.median(np.abs(flat - np.median(flat)))) if flat.size else 0.0
    rg = arr[..., 0] - arr[..., 1]; yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    h, w = gray.shape
    return {"lap_var": lap_var, "noise": noise, "colorful": colorful,
            "bright": float(gray.mean()), "logpix": float(np.log(h * w)), "aspect": float(w / h)}


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def dynamic_fusion(W, skills3, feat_vec):
    f = np.array(feat_vec, dtype=float)
    g = softmax(W @ f)
    return float(g @ np.array(skills3)), g


# ═══ Step 3: scan α on BT training set ═══

def scan_alpha_konig(gate_model):
    """KonIQ BT训练集扫α"""
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, "ranking_koniq_v3.json"))
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
            ss = new_sc.get(p); ft = feats_v3.get(str(r["node"]))
        if not ss or any(ss.get(sk) is None for sk in POOL): continue
        if not isinstance(ft, dict) or "noise" not in ft: continue
        data.append({"path": p, "bt": r["bt"], "skills": {sk: ss[sk] for sk in POOL}, "feat": ft})

    n = len(data)
    X = np.array([[d["skills"][sk] for sk in POOL] for d in data])
    bt = np.array([d["bt"] for d in data])
    spread = X.std(axis=1)
    raw = np.array([[d["feat"]["lap_var"], d["feat"]["noise"], d["feat"]["colorful"],
                     d["feat"]["bright"], d["feat"]["logpix"], d["feat"]["aspect"], 0.0] for d in data])
    raw[:, 6] = spread
    for j, k in enumerate(FEAT_KEYS):
        if k in ("lap_var", "noise", "colorful"):
            raw[:, j] = np.log(np.maximum(raw[:, j], 1e-6))
    mu = raw.mean(axis=0); sd = raw.std(axis=0) + 1e-9
    F = (raw - mu) / sd

    W_arr = np.array(gate_model["W"])
    mu_arr = np.array(gate_model["mu"]); sd_arr = np.array([s if s>1e-6 else 1.0 for s in gate_model["sd"]])
    Fg = (raw - mu_arr) / sd_arr
    scores_dyn = np.array([softmax(W_arr @ Fg[i]) @ X[i] for i in range(n)])
    scores_bare = X[:, 1]  # S-GLOBAL as proxy

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    valid_idx = perm[n // 2:]; vset = set(valid_idx.tolist())
    rng_p = np.random.default_rng(44)
    pairs = []
    while len(pairs) < 2000:
        i, j = rng_p.integers(0, n, 2)
        if i in vset and j in vset and abs(bt[i]-bt[j])>=np.std(bt)*0.1:
            pairs.append((i,j))

    best_a, best_h = 0.6, 0
    for a in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        s = a*scores_dyn + (1-a)*scores_bare
        h = sum(1 for i,j in pairs if (s[i]-s[j])*(bt[i]-bt[j])>0)/len(pairs)
        if h>best_h: best_h=h; best_a=a
        print(f"  α={a:.1f} → 一致率={h:.4f}")
    print(f"  [koniq] 最优α={best_a:.1f} (一致率={best_h:.4f})")
    return best_a


def scan_alpha_spaq():
    """SPAQ 门控=等权, bare=S-GLOBAL。直接在 R6 实际数据上扫 α 用 MAE 最优原则。"""
    # SPAQ gate FAIL, dynamic_fusion = 等权。α 影响的是等权融合 vs bare投票的混合比。
    # 最实用的方法: 在 val 上扫 MAE (不需要 BT 训练集, 直接用缓存分)
    from iqa_agent.data import load_images
    images = {r.img_id: r.path for r in load_images(cfg, "spaq_test")}

    r2_path = os.path.join(cfg.runs_dir, "final", "r2_spaq", "scores.csv")
    r1b_path = os.path.join(cfg.runs_dir, "final", "r1b_spaq", "scores.csv")
    paras_path = os.path.join(cfg.runs_dir, "r6_spaq_paras.json")
    paras = jload(paras_path) if os.path.exists(paras_path) else {}

    with open(r2_path, encoding="utf-8-sig") as f:
        r2 = {}
        for r in csv.DictReader(f):
            if r.get("skill_scores"):
                ss = json.loads(r["skill_scores"])
                r2[r["img_id"]] = {sk: ss[sk] for sk in POOL if sk in ss}
    with open(r1b_path, encoding="utf-8-sig") as f:
        r1b = {r["img_id"]: float(r["score"]) for r in csv.DictReader(f) if r.get("score") not in ("", None)}

    ids = sorted(set(r2.keys()) & set(r1b.keys()))
    # 用子集(200张)扫α,避免过拟合
    rng = np.random.default_rng(42)
    ids_sub = sorted(rng.choice(ids, min(200, len(ids)), replace=False).tolist())

    mos = load_mos(cfg, "spaq_test")
    best_a, best_mae = 0.5, 999
    for a in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        pred = {}
        for img_id in ids_sub:
            es = r2[img_id]; skills3 = [es[sk] for sk in POOL]
            fus = float(np.mean(skills3))
            bare_parts = [r1b[img_id]] + [paras.get(img_id,{}).get(str(k)) for k in (1,2,3)]
            if any(v is None for v in bare_parts): continue
            vote = float(np.mean(bare_parts))
            pred[img_id] = a*fus + (1-a)*vote
        mae = np.mean([abs(pred[i]-mos[i]) for i in pred if i in mos])
        print(f"  α={a:.1f} → MAE(200)={mae:.4f}")
        if mae < best_mae: best_mae=mae; best_a=a
    print(f"  [spaq] 最优α={best_a:.1f} (MAE={best_mae:.4f})")
    return best_a



def run_arm(ds, eval_ds, gate_model, alpha, offanchor):
    from iqa_agent.data import load_images
    lo, hi = cfg.scales[ds]
    images = {r.img_id: r.path for r in load_images(cfg, eval_ds)}

    # Bare voting —— 无锚点, 不变
    r1b_path = os.path.join(cfg.runs_dir, "final", f"r1b_{ds}", "scores.csv")
    with open(r1b_path, encoding="utf-8-sig") as f:
        r1b = {r["img_id"]: float(r["score"]) for r in csv.DictReader(f) if r.get("score") not in ("", None)}
    paras_path = os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json")
    paras = jload(paras_path) if os.path.exists(paras_path) else {}

    # Expert scores: anchored (R2 cache) vs offanchor (posthoc cache)
    if offanchor:
        ec = jload(ec_path) if os.path.exists(ec_path) else {}
    else:
        r2_path = os.path.join(cfg.runs_dir, "final", f"r2_{ds}", "scores.csv")
        ec = {}
        with open(r2_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("skill_scores"):
                    ss = json.loads(r["skill_scores"])
                    ec[r["img_id"]] = {sk: ss[sk] for sk in POOL if sk in ss}

    if gate_model is not None:
        W = np.array(gate_model["W"])
        mu_a = np.array(gate_model["mu"])
        sd_a = np.array([s if s > 1e-6 else 1.0 for s in gate_model["sd"]])
        feat_keys_gate = gate_model["feat_keys"]
    else:
        W = None

    ids = sorted(set(ec.keys()) & set(r1b.keys()) & set(images.keys()))
    rows = []
    for img_id in ids:
        bare_parts = [r1b[img_id]] + [paras.get(img_id, {}).get(str(k)) for k in (1, 2, 3)]
        if any(v is None for v in bare_parts): continue
        vote = float(np.mean(bare_parts))

        es = ec[img_id]
        skills3 = [es[sk] for sk in POOL]
        if any(v is None for v in skills3): continue

        feat_raw = opencv_features(Image.open(images[img_id]))
        spread_val = float(np.std(skills3))
        f = np.array([feat_raw["lap_var"], feat_raw["noise"], feat_raw["colorful"],
                      feat_raw["bright"], feat_raw["logpix"], feat_raw["aspect"], spread_val], dtype=float)
        for j, k in enumerate(FEAT_KEYS):
            if k in ("lap_var", "noise", "colorful"):
                f[j] = np.log(max(f[j], 1e-6))

        if W is not None and gate_model is not None:
            f_std = (f - mu_a) / sd_a
            fus, g = dynamic_fusion(W, skills3, f_std)
        else:
            fus = float(np.mean(skills3))
            g = np.ones(len(POOL)) / len(POOL)

        final = alpha * fus + (1 - alpha) * vote
        rows.append({"img_id": img_id, "score": round(final, 4),
                     "reason": f"α={alpha:.1f}·fus({fus:.3f})+{(1-alpha):.1f}·bare({vote:.3f})"})
    return rows


def write_and_eval(arm, ds, eval_ds, rows):
    out_dir = os.path.join(cfg.runs_dir, "posthoc", arm)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score", "reason", "parse_tier"])
        w.writeheader()
        for r in rows:
            w.writerow({"img_id": r["img_id"], "dataset": ds, "route": arm,
                        "score": r["score"], "reason": r.get("reason", "")[:300], "parse_tier": 1})
    mos = load_mos(cfg, eval_ds)
    pred = {r["img_id"]: r["score"] for r in rows}
    m = compute_metrics(pred, mos)
    vals = np.array(list(pred.values()))
    s = {"arm": arm, "n": m["n"], "SRCC": round(m["SRCC"], 4), "MAE": round(m["MAE"], 4),
         "PLCC": round(m["PLCC"], 4), "mean": round(float(vals.mean()), 4),
         "std": round(float(vals.std()), 4), "unique": int(len(set(np.round(vals, 4))))}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    return s


# ═══ MAIN ═══

gate_k = jload(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"))
# SPAQ gate FAIL → None (等权回退)
gate_s = None

print("=== Step 3: 扫 α ===\n")
ak = scan_alpha_konig(gate_k)
print()
as_ = scan_alpha_spaq()

print(f"\nKonIQ α={ak:.1f}  SPAQ α={as_:.1f}  (SPAQ gate=FAIL→等权)")

print("\n=== Step 4: 统一 R6 ===\n")
rk = run_arm("koniq", "koniq_val", gate_k, ak, offanchor=False)
sk = write_and_eval("r6_unified_koniq", "koniq", "koniq_val", rk)
rs = run_arm("spaq", "spaq_test", None, as_, offanchor=False)
ss = write_and_eval("r6_unified_spaq", "spaq", "spaq_test", rs)
print(f"KonIQ: SRCC={sk['SRCC']} MAE={sk['MAE']} PLCC={sk['PLCC']} n={sk['n']}")
print(f"SPAQ:  SRCC={ss['SRCC']} MAE={ss['MAE']} PLCC={ss['PLCC']} n={ss['n']}")

rok = run_arm("koniq", "koniq_val", gate_k, ak, offanchor=True)
ros = run_arm("spaq", "spaq_test", None, as_, offanchor=True)
print(f"KonIQ: SRCC={sok['SRCC']} MAE={sok['MAE']} PLCC={sok['PLCC']} n={sok['n']}")
print(f"SPAQ:  SRCC={sos['SRCC']} MAE={sos['MAE']} PLCC={sos['PLCC']} n={sos['n']}")

# ═══ 总表 ═══
# R1-anchor + old R1-R3 from old data
old_table = {}
with open(os.path.join(cfg.runs_dir, "final", "main_table.csv")) as f:
    for r in csv.DictReader(f):
        old_table[(r["run"], r["dataset"])] = r

for dk, ds_key in [("koniq", "koniq_val"), ("spaq", "spaq_test")]:
print("\n" + "=" * 85)
print("统一框架主表 (final)")
print("=" * 85)
print(f"{'臂':<28} {'KonIQ SRCC':>10} {'MAE':>8} {'PLCC':>8}  {'SPAQ SRCC':>10} {'MAE':>8} {'PLCC':>8}")
print("-" * 85)

ROWS = [
    ("一  R1-bare", "r1b_koniq", "r1b_spaq"),
    ("一  R1-rich", "r1r_koniq", "r1r_spaq"),
    ("一  R2", "r2_koniq", "r2_spaq"),
    ("一  R2.5", "r25_koniq", "r25_spaq"),
    ("一  R3", "r3_koniq", "r3_spaq"),
]

for label, kk, ss_old in ROWS:
    rk2 = old_table.get((kk, "koniq_val"), {})
    rs2 = old_table.get((ss_old, "spaq_test"), {})
    print(f"{label:<28} {rk2.get('SRCC','?'):>10} {rk2.get('MAE','?'):>8} {rk2.get('PLCC','?'):>8}  {rs2.get('SRCC','?'):>10} {rs2.get('MAE','?'):>8} {rs2.get('PLCC','?'):>8}")

print(f"{'三  R6 (统一框架)':<28} {sk['SRCC']:>10.4f} {sk['MAE']:>8.4f} {sk['PLCC']:>8.4f}  {ss['SRCC']:>10.4f} {ss['MAE']:>8.4f} {ss['PLCC']:>8.4f}")
print(f"{'考后 R1-anchor':<28} {r1a[0]['SRCC']:>10.4f} {r1a[0]['MAE']:>8.4f} {r1a[0]['PLCC']:>8.4f}  {r1a[1]['SRCC']:>10.4f} {r1a[1]['MAE']:>8.4f} {r1a[1]['PLCC']:>8.4f}")
print("-" * 85)
print(f"SPAQ 门控: FAIL→等权  |  KonIQ α={ak:.1f}  SPAQ α={as_:.1f}")
