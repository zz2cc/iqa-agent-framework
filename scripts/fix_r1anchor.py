# -*- coding: utf-8 -*-
"""R1-anchor 修复版：轻专家 + 锚点分数带 + 文字五级描述 + 展宽指令。
砍掉：维度定义、检查清单、评估程序、JSON输出契约。输出只回一个数。"""
import asyncio, csv, json, os, sys, time, numpy as np
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics

# ── prompt模板 ──
TEMPLATES = {
    "koniq": """You are an image quality assessor. Rate the overall quality of this image on a scale from 1.0 to 5.0.

[Quality levels]
1 = Bad; unacceptable quality.
2 = Poor; flaws are annoying and hard to ignore.
3 = Fair; flaws are noticeable but tolerable.
4 = Good; minor flaws, easy to overlook.
5 = Excellent overall impression; nothing to complain about.

[Score scale]
The final score is a float in [1, 5]. Level-to-band guide:
  5 Excellent -> 4.3-5.0 | 4 Good -> 3.5-4.2 | 3 Fair -> 2.5-3.4 | 2 Poor -> 1.5-2.4 | 1 Bad -> 1.0-1.4
Use the FULL range; do not cluster scores near the middle.

Reply with only a single number.""",

    "spaq": """You are an image quality assessor. Rate the overall quality of this image on a scale from 0.0 to 10.0.

[Quality levels]
1 = Bad; unacceptable quality.
2 = Poor; flaws are annoying and hard to ignore.
3 = Fair; flaws are noticeable but tolerable.
4 = Good; minor flaws, easy to overlook.
5 = Excellent overall impression; nothing to complain about.

[Score scale]
The final score is a float in [0, 10]. Level-to-band guide:
  5 Excellent -> 8.5-10 | 4 Good -> 6.5-8.4 | 3 Fair -> 4.5-6.4 | 2 Poor -> 2.5-4.4 | 1 Bad -> 0-2.4
Use the FULL range; do not cluster scores near the middle.

Reply with only a single number.""",
}


async def main():
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)

    for ds, eval_ds in [("koniq", "koniq_val"), ("spaq", "spaq_test")]:
        lo, hi = cfg.scales[ds]
        images = load_images(cfg, eval_ds)
        prompt = TEMPLATES[ds]

        print(f"\n=== {ds.upper()} R1-anchor 修复版 ===")
        print(f"  Prompt ({len(prompt)} chars)")
        print(f"  图片数: {len(images)}")
        t0 = time.time()

        async def one(img):
            text, _ = await client.score_image(img.path, prompt, temperature=0.0)
            p = parse_score(text, (lo, hi))
            return img.img_id, p["score"] if p else None

        jobs = [one(img) for img in images]
        raw = await gather_with_progress(jobs, every=200, label=f"r1a-fix-{ds}")

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
        print(f"  MOS均值={np.mean(list(mos.values())):.4f}")

        od = os.path.join(cfg.runs_dir, "posthoc", f"r1anchor_v2_{ds}")
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "scores.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score", "reason", "parse_tier"])
            w.writeheader()
            for iid, s in pred.items():
                w.writerow({"img_id": iid, "dataset": ds, "route": "r1anchor_v2", "score": round(s, 4), "reason": "", "parse_tier": 1})

        s = {"arm": f"r1anchor_v2_{ds}", "n": m["n"], "SRCC": round(m["SRCC"], 4),
             "MAE": round(m["MAE"], 4), "PLCC": round(m["PLCC"], 4),
             "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
             "unique": int(len(set(np.round(vals, 4))))}
        with open(os.path.join(od, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)

    print(f"\n账本: {client.ledger()}")


if __name__ == "__main__":
    asyncio.run(main())
