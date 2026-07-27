# -*- coding: utf-8 -*-
"""R6 hybrid: R3经验规则版 S-TECH/S-GLOBAL + R2 S-CONTENT + 32B W + bare投票。零 API。"""
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

fk = json.load(open(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json")))
W32 = np.array(fk["W"]); mu32 = np.array(fk["mu"]); sd32 = np.array([s if s > 1e-6 else 1.0 for s in fk["sd"]])


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


# ── 加载数据 ──
r2 = {}
with open(os.path.join(cfg.runs_dir, "final", "r2_koniq", "scores.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("skill_scores"): r2[r["img_id"]] = json.loads(r["skill_scores"])

r3 = {}
with open(os.path.join(cfg.runs_dir, "final", "r3_koniq", "scores.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("skill_scores"): r3[r["img_id"]] = json.loads(r["skill_scores"])

r1b = {}
with open(os.path.join(cfg.runs_dir, "final", "r1b_koniq", "scores.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("score") not in ("", None): r1b[r["img_id"]] = float(r["score"])

paras = json.load(open(os.path.join(cfg.runs_dir, "r6_koniq_paras.json")))
images = {r.img_id: r.path for r in load_images(cfg, "koniq_val")}

# ── 组装混合专家分 ──
n_r3_t, n_r3_g = 0, 0
expert = {}
for iid in r2:
    t = r3.get(iid, {}).get("S-TECH")
    g = r3.get(iid, {}).get("S-GLOBAL")
    if t is not None: n_r3_t += 1
    else: t = r2[iid].get("S-TECH")
    if g is not None: n_r3_g += 1
    else: g = r2[iid].get("S-GLOBAL")
    c = r2[iid].get("S-CONTENT")
    if t is not None and g is not None and c is not None:
        expert[iid] = {"S-TECH": t, "S-GLOBAL": g, "S-CONTENT": c}

print(f"R3 S-TECH: {n_r3_t}/{len(r2)}  R3 S-GLOBAL: {n_r3_g}/{len(r2)}")
print(f"混合 Expert: {len(expert)} 张")

# ── 融合 ──
ids = sorted(set(expert.keys()) & set(r1b.keys()) & set(paras.keys()) & set(images.keys()))
rows = []
for iid in ids:
    es = expert[iid]; s3 = [es[sk] for sk in POOL]
    sp = float(np.std(s3))
    img = Image.open(images[iid])
    feat_raw = opencv_features(img)
    f = np.array([feat_raw["lap_var"], feat_raw["noise"], feat_raw["colorful"],
                  feat_raw["bright"], feat_raw["logpix"], feat_raw["aspect"], sp], dtype=float)
    for j, k in enumerate(FEAT_KEYS):
        if k in ("lap_var", "noise", "colorful"): f[j] = np.log(max(f[j], 1e-6))
    g = softmax(W32 @ ((f - mu32) / sd32))
    fus = float(g @ np.array(s3))
    bp = [r1b[iid]] + [paras.get(iid, {}).get(str(k)) for k in (1, 2, 3)]
    if any(v is None for v in bp): continue
    vote = float(np.mean(bp))
    rows.append({"img_id": iid, "score": round(0.6 * fus + 0.4 * vote, 4)})

mos = load_mos(cfg, "koniq_val")
pred = {r["img_id"]: r["score"] for r in rows}
m = compute_metrics(pred, mos)
vals = np.array(list(pred.values()))

print(f"SRCC={m['SRCC']:.4f}  MAE={m['MAE']:.4f}  PLCC={m['PLCC']:.4f}")
print(f"均值={vals.mean():.4f}  std={vals.std():.4f}  唯一值={len(set(np.round(vals, 4)))}")
print(f"n={m['n']}  新 API: 0")
