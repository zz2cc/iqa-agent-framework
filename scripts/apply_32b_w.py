# -*- coding: utf-8 -*-
"""将 32B W 矩阵应用到 8B 专家分。全缓存命中，0 API。"""
import asyncio, csv, json, os, sys, numpy as np
from PIL import Image
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.prompts.skills import build_skill_prompt
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics

cfg = get_config()
POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]

fk = json.load(open(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json")))
W32 = np.array(fk["W"]); mu32 = np.array(fk["mu"]); sd32 = np.array([s if s > 1e-6 else 1.0 for s in fk["sd"]])

B4 = {
    "koniq": [
        "Rate the overall quality of this image on a scale from 1 to 5. Reply with only a single number.",
        "On a scale from 1 to 5, how would you rate the overall quality of this image? Reply with only a single number.",
        "Give a single overall quality score for this image, from 1 (worst) to 5 (best). Reply with only the number.",
        "As an image quality rater, assign one overall quality score from 1 to 5 to this image. Reply with only a single number.",
    ],
    "spaq": [
        "Rate the overall quality of this image on a scale from 0 to 10. Reply with only a single number.",
        "On a scale from 0 to 10, how would you rate the overall quality of this image? Reply with only a single number.",
        "Give a single overall quality score for this image, from 0 (worst) to 10 (best). Reply with only the number.",
        "As an image quality rater, assign one overall quality score from 0 to 10 to this image. Reply with only a single number.",
    ],
}


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


async def main():
    client = VLMClient(cfg, cfg.model_debug)

    for ds, eval_ds, alpha in [("koniq", "koniq_val", 0.6)]:
        lo, hi = cfg.scales[ds]
        images = load_images(cfg, eval_ds)
        id_to_path = {img.img_id: img.path for img in images}

        # 8B R1-bare (already cached from probe)
        r1b = {}
        with open(os.path.join(cfg.runs_dir, "posthoc", f"r1b_8b_{ds}", "scores.csv"), encoding="utf-8-sig") as f:
            for r in csv.DictReader(f): r1b[r["img_id"]] = float(r["score"])

        # Rebuild 8B expert + para scores from SHA256 cache (0 API)
        print(f"[{ds}] 重建8B专家分(缓存命中)...")
        expert = {}
        for sk in POOL:
            prompt = build_skill_prompt(sk, ds)
            async def one_exp(img, prompt=prompt):
                text, _ = await client.score_image(img.path, prompt, temperature=0.0)
                p = parse_score(text, (lo, hi))
                return img.img_id, p["score"] if p else None
            raw = await gather_with_progress([one_exp(img) for img in images], every=500, label=f"re-{sk[:4]}")
            for r in raw:
                if not isinstance(r, Exception) and r[1] is not None:
                    expert.setdefault(r[0], {})[sk] = r[1]

        print(f"[{ds}] 重建8B bare释义(缓存命中)...")
        paras = {}
        for k in (1, 2, 3):
            prompt = B4[ds][k]
            async def one_p(img, prompt=prompt):
                try:
                    text, _ = await client.score_image(img.path, prompt, temperature=0.0)
                    p = parse_score(text, (lo, hi))
                    return img.img_id, p["score"] if p else None
                except Exception:
                    return img.img_id, None
            raw = await gather_with_progress([one_p(img) for img in images], every=500, label=f"re-p{k}")
            for r in raw:
                if not isinstance(r, Exception) and r is not None and r[1] is not None:
                    paras.setdefault(r[0], {})[str(k)] = r[1]

        # Apply 32B W
        print(f"[{ds}] 应用32B W矩阵...")
        common = sorted(set(expert.keys()) & set(r1b.keys()) & set(paras.keys()) & set(id_to_path.keys()))
        rows = []
        for iid in common:
            es = expert[iid]; s3 = [es[sk] for sk in POOL]
            if any(v is None for v in s3): continue
            sp = float(np.std(s3))
            img = Image.open(id_to_path[iid])
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
            rows.append({"img_id": iid, "score": round(alpha * fus + (1 - alpha) * vote, 4)})

        mos = load_mos(cfg, eval_ds)
        pred = {r["img_id"]: r["score"] for r in rows}
        m = compute_metrics(pred, mos)
        vals = np.array(list(pred.values()))
        print(f"  SRCC={m['SRCC']:.4f}  MAE={m['MAE']:.4f}  PLCC={m['PLCC']:.4f}")
        print(f"  均值={vals.mean():.4f}  std={vals.std():.4f}  唯一值={len(set(np.round(vals, 4)))}")
        print(f"  账本: {client.ledger()}")


if __name__ == "__main__":
    asyncio.run(main())
