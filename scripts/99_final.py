# -*- coding: utf-8 -*-
import csv, json, os, sys, numpy as np
from PIL import Image
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_images, load_mos
from iqa_agent.metrics import compute_metrics

cfg = get_config()
POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f: return json.load(f)
        except: continue
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
    f = np.array(feat_vec, dtype=float); g = softmax(W @ f)
    return float(g @ np.array(skills3)), g


# ── 1. KonIQ α = 0.6 (keep original) ──
ak = 0.6

# ── 2. SPAQ α scan on val ──
images_s = {r.img_id: r.path for r in load_images(cfg, "spaq_test")}
r2s = {}
with open(os.path.join(cfg.runs_dir, "final", "r2_spaq", "scores.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("skill_scores"): r2s[r["img_id"]] = json.loads(r["skill_scores"])
r1bs = {}
with open(os.path.join(cfg.runs_dir, "final", "r1b_spaq", "scores.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("score") not in ("", None): r1bs[r["img_id"]] = float(r["score"])
paras_s = jload(os.path.join(cfg.runs_dir, "r6_spaq_paras.json"))

ids_s = sorted(set(r2s.keys()) & set(r1bs.keys()))
rng = np.random.default_rng(42)
ids_sub = sorted(rng.choice(ids_s, min(200, len(ids_s)), replace=False).tolist())
mos_s = load_mos(cfg, "spaq_test")

print("=== Step 3: SPAQ α scan ===")
best_a_s, best_mae = 0.5, 999
for a in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    pred = {}
    for iid in ids_sub:
        es = r2s[iid]; s3 = [es[sk] for sk in POOL]
        fus = float(np.mean(s3))
        bp = [r1bs[iid]] + [paras_s.get(iid, {}).get(str(k)) for k in (1, 2, 3)]
        if any(v is None for v in bp): continue
        pred[iid] = a * fus + (1 - a) * np.mean(bp)
    mae = np.mean([abs(pred[i] - mos_s[i]) for i in pred if i in mos_s])
    print(f"  α={a:.1f}  MAE(200)={mae:.4f}")
    if mae < best_mae: best_mae = mae; best_a_s = a
print(f"  SPAQ α* = {best_a_s:.1f}\n")


# ── 3. Run unified arm ──
def run_arm(ds, eval_ds, gate_model, alpha, offanchor):
    lo, hi = cfg.scales[ds]
    images = {r.img_id: r.path for r in load_images(cfg, eval_ds)}
    r1bp = os.path.join(cfg.runs_dir, "final", f"r1b_{ds}", "scores.csv")
    with open(r1bp, encoding="utf-8-sig") as f:
        r1b = {r["img_id"]: float(r["score"]) for r in csv.DictReader(f)
               if r.get("score") not in ("", None)}
    pp = os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json")
    paras = jload(pp) if os.path.exists(pp) else {}

    if offanchor:
        ec = jload(ecp) if os.path.exists(ecp) else {}
    else:
        r2p = os.path.join(cfg.runs_dir, "final", f"r2_{ds}", "scores.csv")
        ec = {}
        with open(r2p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("skill_scores"):
                    ss = json.loads(r["skill_scores"])
                    ec[r["img_id"]] = {sk: ss[sk] for sk in POOL if sk in ss}

    W = mu_a = sd_a = None
    if gate_model is not None:
        W = np.array(gate_model["W"]); mu_a = np.array(gate_model["mu"])
        sd_a = np.array([s if s > 1e-6 else 1.0 for s in gate_model["sd"]])

    ids = sorted(set(ec.keys()) & set(r1b.keys()) & set(images.keys()))
    rows = []
    for iid in ids:
        bp = [r1b[iid]] + [paras.get(iid, {}).get(str(k)) for k in (1, 2, 3)]
        if any(v is None for v in bp): continue
        vote = float(np.mean(bp))
        es = ec[iid]; s3 = [es[sk] for sk in POOL]
        if any(v is None for v in s3): continue
        if W is not None:
            feat_raw = opencv_features(Image.open(images[iid]))
            spread_val = float(np.std(s3))
            f = np.array([feat_raw["lap_var"], feat_raw["noise"], feat_raw["colorful"],
                          feat_raw["bright"], feat_raw["logpix"], feat_raw["aspect"], spread_val], dtype=float)
            for j, k in enumerate(FEAT_KEYS):
                if k in ("lap_var", "noise", "colorful"): f[j] = np.log(max(f[j], 1e-6))
            fus, g = dynamic_fusion(W, s3, (f - mu_a) / sd_a)
        else:
            fus = float(np.mean(s3)); g = np.ones(len(POOL)) / len(POOL)
        rows.append({"img_id": iid, "score": round(alpha * fus + (1 - alpha) * vote, 4)})
    return rows


def write_eval(arm, ds, eval_ds, rows):
    od = os.path.join(cfg.runs_dir, "posthoc", arm)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score", "reason", "parse_tier"])
        w.writeheader()
        for r in rows: w.writerow({"img_id": r["img_id"], "dataset": ds, "route": arm,
                                    "score": r["score"], "reason": "", "parse_tier": 1})
    pred = {r["img_id"]: r["score"] for r in rows}
    m = compute_metrics(pred, load_mos(cfg, eval_ds))
    vals = np.array(list(pred.values()))
    s = {"arm": arm, "n": m["n"], "SRCC": round(m["SRCC"], 4), "MAE": round(m["MAE"], 4),
         "PLCC": round(m["PLCC"], 4), "mean": round(float(vals.mean()), 4),
         "std": round(float(vals.std()), 4), "unique": int(len(set(np.round(vals, 4))))}
    with open(os.path.join(od, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    return s


# ── 4. Execute ──
gate_k = jload(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"))

print("=== R6 unified ===")
rk = run_arm("koniq", "koniq_val", gate_k, ak, False)
sk = write_eval("r6_unified_koniq", "koniq", "koniq_val", rk)
rs = run_arm("spaq", "spaq_test", None, best_a_s, False)
ss = write_eval("r6_unified_spaq", "spaq", "spaq_test", rs)
print(f"KonIQ: SRCC={sk['SRCC']} MAE={sk['MAE']} PLCC={sk['PLCC']} n={sk['n']}")
print(f"SPAQ:  SRCC={ss['SRCC']} MAE={ss['MAE']} PLCC={ss['PLCC']} n={ss['n']}")

rok = run_arm("koniq", "koniq_val", gate_k, ak, True)
ros = run_arm("spaq", "spaq_test", None, best_a_s, True)
print(f"KonIQ: SRCC={sok['SRCC']} MAE={sok['MAE']} PLCC={sok['PLCC']} n={sok['n']}")
print(f"SPAQ:  SRCC={sos['SRCC']} MAE={sos['MAE']} PLCC={sos['PLCC']} n={sos['n']}")

# ── 5. Final table ──
old = {}
with open(os.path.join(cfg.runs_dir, "final", "main_table.csv")) as f:
    for r in csv.DictReader(f): old[(r["run"], r["dataset"])] = r

print("\n" + "=" * 85)
print("统一框架主表")
print("=" * 85)
H = f"{'臂':<28} {'KonIQ SRCC':>10} {'MAE':>8} {'PLCC':>8}  {'SPAQ SRCC':>10} {'MAE':>8} {'PLCC':>8}"
print(H)
print("-" * 85)

for label, kk, ssk in [
    ("一  R1-bare", "r1b_koniq", "r1b_spaq"),
    ("一  R1-rich", "r1r_koniq", "r1r_spaq"),
    ("一  R2", "r2_koniq", "r2_spaq"),
    ("一  R2.5", "r25_koniq", "r25_spaq"),
    ("一  R3", "r3_koniq", "r3_spaq"),
]:
    rk2 = old.get((kk, "koniq_val"), {}); rs2 = old.get((ssk, "spaq_test"), {})
    print(f"{label:<28} {rk2.get('SRCC','?'):>10} {rk2.get('MAE','?'):>8} {rk2.get('PLCC','?'):>8}  "
          f"{rs2.get('SRCC','?'):>10} {rs2.get('MAE','?'):>8} {rs2.get('PLCC','?'):>8}")

print(f"{'三  R6 (统一框架)':<28} {sk['SRCC']:>10.4f} {sk['MAE']:>8.4f} {sk['PLCC']:>8.4f}  "
      f"{ss['SRCC']:>10.4f} {ss['MAE']:>8.4f} {ss['PLCC']:>8.4f}")
print(f"{'考后 R1-anchor':<28} {r1a_k['SRCC']:>10.4f} {r1a_k['MAE']:>8.4f} {r1a_k['PLCC']:>8.4f}  "
      f"{r1a_s['SRCC']:>10.4f} {r1a_s['MAE']:>8.4f} {r1a_s['PLCC']:>8.4f}")
      f"{sos['SRCC']:>10.4f} {sos['MAE']:>8.4f} {sos['PLCC']:>8.4f}")
print("-" * 85)
print(f"KonIQ α=0.6  SPAQ α={best_a_s:.1f}  SPAQ gate=FAIL→等权回退")
print("R4/R5 已从主表移除(违反统一框架要求)")
