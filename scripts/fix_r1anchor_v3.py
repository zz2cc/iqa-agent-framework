# -*- coding: utf-8 -*-
"""R1-anchor v3: 轻专家 + 锚点带 + 文字等级 + 维度 + 清单 + 程序 + JSON。
唯一跟 R1-rich 的差异: 人设是轻量的 "image quality assessor", 不加 "expert" 和 "Holistic Quality Assessor" """
import asyncio, csv, json, os, sys, time, numpy as np
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics

TEMPLATES = {
    "koniq": """You are an image quality assessor.

[Your dimension]
You assess the OVERALL quality impression exactly as a typical viewer would, considering everything together without breaking it into dimensions.

[Checklist — inspect each aspect]
- First impression: Immediate gut reaction: good or bad?
- Anything annoying: Any single issue that dominates your impression?
- Overall acceptability: Would an ordinary viewer find this image acceptable, nice, or defective?

[Quality levels]
1 = Bad; unacceptable quality.
2 = Poor; flaws are annoying and hard to ignore.
3 = Fair; flaws are noticeable but tolerable.
4 = Good; minor flaws, easy to overlook.
5 = Excellent overall impression; nothing to complain about.

[Assessment procedure — follow strictly]
1. Scan the whole image for a first impression.
2. Inspect each aspect in the checklist below, one by one.
3. Identify the DOMINANT quality issue(s) for your dimension.
4. Judge their severity objectively.
5. Map your judgment to a level, then give a precise score within that level's band.

[Score scale]
The final score is a float in [1, 5]. Level-to-band guide:
  5 Excellent -> 4.3-5.0 | 4 Good -> 3.5-4.2 | 3 Fair -> 2.5-3.4 | 2 Poor -> 1.5-2.4 | 1 Bad -> 1.0-1.4
Use the FULL range; do not cluster scores near the middle.

[Output format — strict]
Reply with JSON only, no other text:
{"level": <int 1-5>, "score": <float>, "reason": "<= 25 words, key evidence only"}""",

    "spaq": """You are an image quality assessor.

[Your dimension]
You assess the OVERALL quality impression exactly as a typical viewer would, considering everything together without breaking it into dimensions.

[Checklist — inspect each aspect]
- First impression: Immediate gut reaction: good or bad?
- Anything annoying: Any single issue that dominates your impression?
- Overall acceptability: Would an ordinary viewer find this image acceptable, nice, or defective?

[Quality levels]
1 = Bad; unacceptable quality.
2 = Poor; flaws are annoying and hard to ignore.
3 = Fair; flaws are noticeable but tolerable.
4 = Good; minor flaws, easy to overlook.
5 = Excellent overall impression; nothing to complain about.

[Assessment procedure — follow strictly]
1. Scan the whole image for a first impression.
2. Inspect each aspect in the checklist below, one by one.
3. Identify the DOMINANT quality issue(s) for your dimension.
4. Judge their severity objectively.
5. Map your judgment to a level, then give a precise score within that level's band.

[Score scale]
The final score is a float in [0, 10]. Level-to-band guide:
  5 Excellent -> 8.5-10 | 4 Good -> 6.5-8.4 | 3 Fair -> 4.5-6.4 | 2 Poor -> 2.5-4.4 | 1 Bad -> 0-2.4
Use the FULL range; do not cluster scores near the middle.

[Output format — strict]
Reply with JSON only, no other text:
{"level": <int 1-5>, "score": <float>, "reason": "<= 25 words, key evidence only"}""",
}


async def main():
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)

    for ds, eval_ds in [("koniq", "koniq_val"), ("spaq", "spaq_test")]:
        lo, hi = cfg.scales[ds]
        images = load_images(cfg, eval_ds)
        prompt = TEMPLATES[ds]
        t0 = time.time()

        print(f"\n=== {ds.upper()} R1-anchor v3 ===")
        print(f"  Prompt: {len(prompt)} chars, {len(images)} images")

        async def one(img):
            text, _ = await client.score_image(img.path, prompt, temperature=0.0)
            p = parse_score(text, (lo, hi))
            return img.img_id, p["score"] if p else None

        raw = await gather_with_progress([one(img) for img in images], every=200, label=f"r1a-v3-{ds}")

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

        od = os.path.join(cfg.runs_dir, "posthoc", f"r1anchor_v3_{ds}")
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "scores.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score", "reason", "parse_tier"])
            w.writeheader()
            for iid, s in pred.items():
                w.writerow({"img_id": iid, "dataset": ds, "route": "r1anchor_v3", "score": round(s, 4), "reason": "", "parse_tier": 1})

        json.dump({"arm": f"r1anchor_v3_{ds}", "n": m["n"], "SRCC": round(m["SRCC"], 4),
                   "MAE": round(m["MAE"], 4), "PLCC": round(m["PLCC"], 4),
                   "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
                   "unique": int(len(set(np.round(vals, 4))))},
                  open(os.path.join(od, "summary.json"), "w"), ensure_ascii=False, indent=2)

    print(f"\n账本: {client.ledger()}")


if __name__ == "__main__":
    asyncio.run(main())
