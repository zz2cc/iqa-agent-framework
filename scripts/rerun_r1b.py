# -*- coding: utf-8 -*-
"""绕过缓存重新跑R1-bare双域"""
import asyncio, os, sys, time, numpy as np
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics


async def main():
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)

    for ds, eval_ds in [("koniq", "koniq_val"), ("spaq", "spaq_test")]:
        lo, hi = cfg.scales[ds]
        images = load_images(cfg, eval_ds)
        prompt = f"Rate the overall quality of this image on a scale from {lo} to {hi}. Reply with only a single number."

        print(f"\n=== {ds.upper()} R1-bare 重跑 ===")
        print(f"  图片数: {len(images)}  量程: {lo}-{hi}")
        print(f"  Prompt: \"{prompt}\"")

        t0 = time.time()

        async def one(img):
            text, _ = await client.score_image(img.path, prompt, temperature=0.0)
            p = parse_score(text, (lo, hi))
            return img.img_id, (p["score"] if p else None)

        jobs = [one(img) for img in images]
        results = await gather_with_progress(jobs, every=200, label=f"r1b-{ds}")

        pred = {}
        for r in results:
            if not isinstance(r, Exception) and r[1] is not None:
                pred[r[0]] = float(r[1])

        elapsed = time.time() - t0
        mos = load_mos(cfg, eval_ds)
        m = compute_metrics(pred, mos)
        vals = np.array(list(pred.values()))

        print(f"  完成: {len(pred)}/{len(images)} 有效  耗时: {elapsed:.0f}s")
        print(f"  SRCC={m['SRCC']:.4f}  MAE={m['MAE']:.4f}  PLCC={m['PLCC']:.4f}")
        print(f"  均值={vals.mean():.4f}  std={vals.std():.4f}  唯一值={len(set(np.round(vals, 4)))}")
        print(f"  MOS均值={np.mean(list(mos.values())):.4f}  MOS std={np.std(list(mos.values())):.4f}")

    print(f"\n账本: {client.ledger()}")


if __name__ == "__main__":
    asyncio.run(main())
