# -*- coding: utf-8 -*-
"""8B R6 统一框架: 重跑 3专家 + bare释义, 等权融合(无32B的W矩阵), α=0.6/0.3"""
import asyncio, csv, json, os, sys, time, numpy as np
from PIL import Image
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.prompts.skills import build_skill_prompt, build_r1_bare_prompt
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics

cfg = get_config()
POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]

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


async def main():
    client = VLMClient(cfg, cfg.model_debug)
    print(f"模型: {cfg.model_debug}")

    for ds, eval_ds, alpha in [("koniq", "koniq_val", 0.6), ("spaq", "spaq_test", 0.3)]:
        lo, hi = cfg.scales[ds]
        images = load_images(cfg, eval_ds)
        ids_all = sorted([img.img_id for img in images])
        t0 = time.time()

        # ── 3 expert scores ──
        print(f"\n{'='*50}")
        print(f"[{ds.upper()}] 8B 3专家评分")
        expert = {}
        for sk in POOL:
            prompt = build_skill_prompt(sk, ds)
            todo = [(img.img_id, img.path) for img in images]
            print(f"  {sk}: {len(todo)} 张")

            async def one_expert(img_id, img_path, sk=sk, prompt=prompt):
                text, _ = await client.score_image(img_path, prompt, temperature=0.0)
                p = parse_score(text, (lo, hi))
                return img_id, sk, p["score"] if p else None

            jobs = [one_expert(iid, ipath) for iid, ipath in todo]
            raw = await gather_with_progress(jobs, every=200, label=f"8b-{ds}-{sk[:4]}")
            for r in raw:
                if not isinstance(r, Exception) and r[2] is not None:
                    expert.setdefault(r[0], {})[r[1]] = r[2]

        # ── bare para0: reuse 8B probe results (already scored) ──
        print(f"\n[{ds.upper()}] 8B bare释义 para0(复用8B probe)")
        r1b = {}
        with open(os.path.join(cfg.runs_dir, "posthoc", f"r1b_8b_{ds}", "scores.csv"), encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r1b[r["img_id"]] = float(r["score"])

        # ── bare para1-3 ──
        print(f"[{ds.upper()}] 8B bare释义 para1-3")

        paras = {}
        for k in (1, 2, 3):
            prompt = B4[ds][k]
            todo = [(img.img_id, img.path) for img in images]
            print(f"  para{k}: {len(todo)} 张")

            async def one_para(img_id, img_path, prompt=prompt):
                try:
                    text, _ = await client.score_image(img_path, prompt, temperature=0.0)
                    p = parse_score(text, (lo, hi))
                    return img_id, p["score"] if p else None
                except Exception:
                    return img_id, None

            jobs = [one_para(iid, ipath) for iid, ipath in todo]
            raw = await gather_with_progress(jobs, every=200, label=f"8b-{ds}-p{k}")
            for r in raw:
                if not isinstance(r, Exception) and r is not None and r[1] is not None:
                    paras.setdefault(r[0], {})[str(k)] = r[1]

        # ── R6: equal-weight fusion + bare vote ──
        print(f"\n[{ds.upper()}] R6 组装 (等权融合)")
        rows = []
        for img_id in ids_all:
            es = expert.get(img_id, {})
            s3 = [es.get(sk) for sk in POOL]
            if any(v is None for v in s3): continue
            fus = float(np.mean(s3))
            bp = [r1b.get(img_id)] + [paras.get(img_id, {}).get(str(k)) for k in (1, 2, 3)]
            if any(v is None for v in bp): continue
            vote = float(np.mean(bp))
            final = alpha * fus + (1 - alpha) * vote
            rows.append({"img_id": img_id, "score": round(final, 4)})

        elapsed = time.time() - t0
        mos = load_mos(cfg, eval_ds)
        pred = {r["img_id"]: r["score"] for r in rows}
        m = compute_metrics(pred, mos)
        vals = np.array(list(pred.values()))
        print(f"  {len(rows)}/{len(ids_all)}  {elapsed:.0f}s")
        print(f"  SRCC={m['SRCC']:.4f}  MAE={m['MAE']:.4f}  PLCC={m['PLCC']:.4f}")
        print(f"  均值={vals.mean():.4f}  std={vals.std():.4f}  唯一值={len(set(np.round(vals, 4)))}")

        # Save
        od = os.path.join(cfg.runs_dir, "posthoc", f"r6_8b_{ds}")
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "scores.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score", "reason", "parse_tier"])
            w.writeheader()
            for r in rows:
                w.writerow({"img_id": r["img_id"], "dataset": ds, "route": "r6_8b", "score": r["score"], "reason": "", "parse_tier": 1})
        json.dump({"arm": f"r6_8b_{ds}", "n": m["n"], "SRCC": round(m["SRCC"], 4),
                   "MAE": round(m["MAE"], 4), "PLCC": round(m["PLCC"], 4),
                   "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
                   "unique": int(len(set(np.round(vals, 4))))},
                  open(os.path.join(od, "summary.json"), "w"), ensure_ascii=False, indent=2)

    print(f"\n账本: {client.ledger()}")


if __name__ == "__main__":
    asyncio.run(main())
