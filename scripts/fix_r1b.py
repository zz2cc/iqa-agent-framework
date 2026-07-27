# -*- coding: utf-8 -*-
"""修复 R1-bare：浮点措辞(from 1.0 to 5.0) → 重跑 → 更新主表"""
import asyncio, csv, json, os, sys, time, numpy as np
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.prompts.skills import build_r1_bare_prompt
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics


async def main():
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)
    results = {}

    for ds, eval_ds in [("koniq", "koniq_val"), ("spaq", "spaq_test")]:
        lo, hi = cfg.scales[ds]
        images = load_images(cfg, eval_ds)
        prompt = build_r1_bare_prompt(ds)

        print(f"\n=== {ds.upper()} R1-bare 修复版 ===")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  图片数: {len(images)}")

        t0 = time.time()

        async def one(img):
            text, _ = await client.score_image(img.path, prompt, temperature=0.0)
            p = parse_score(text, (lo, hi))
            return img.img_id, p["score"] if p else None

        jobs = [one(img) for img in images]
        raw = await gather_with_progress(jobs, every=200, label=f"r1b-fix-{ds}")

        pred = {}
        for r in raw:
            if not isinstance(r, Exception) and r[1] is not None:
                pred[r[0]] = float(r[1])

        elapsed = time.time() - t0
        mos = load_mos(cfg, eval_ds)
        m = compute_metrics(pred, mos)
        vals = np.array(list(pred.values()))

        print(f"  完成: {len(pred)}/{len(images)}  耗时: {elapsed:.0f}s")
        print(f"  SRCC={m['SRCC']:.4f}  MAE={m['MAE']:.4f}  PLCC={m['PLCC']:.4f}")
        print(f"  均值={vals.mean():.4f}  std={vals.std():.4f}  唯一值={len(set(np.round(vals, 4)))}")

        results[ds] = m

        # Save to posthoc
        od = os.path.join(cfg.runs_dir, "posthoc", f"r1b_fixed_{ds}")
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "scores.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score", "reason", "parse_tier"])
            w.writeheader()
            for iid, s in pred.items():
                w.writerow({"img_id": iid, "dataset": ds, "route": "r1b_fixed", "score": round(s, 4), "reason": "", "parse_tier": 1})

    print(f"\n账本: {client.ledger()}")


if __name__ == "__main__":
    asyncio.run(main())
