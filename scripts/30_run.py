# -*- coding: utf-8 -*-
"""通用推理入口（S2/S7）。

用法：
  python scripts/30_run.py --route r1b --dataset koniq_train --limit 500 --model debug
  python scripts/30_run.py --route r1r --dataset koniq_train --limit 500 --model debug

输出：runs/<route>_<dataset>_<model>_<ts>/scores.csv + meta.json
⚠️ 本脚本绝不读取 MOS（合规：MOS 仅 50_eval.py 使用）。
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iqa_agent.config import get_config
from iqa_agent.client import VLMClient
from iqa_agent.data import load_images
from iqa_agent.pipeline import run_r1, run_r2
from iqa_agent.router import load_router_assets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=["r1b", "r1r", "r2", "r25"])
    ap.add_argument("--dataset", required=True, choices=["koniq_train", "koniq_val", "spaq_test"])
    ap.add_argument("--model", default="debug", choices=["main", "debug"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="抽样种子（默认用 config.seed）")
    ap.add_argument("--library", default=None, help="R3 用：经验库 JSON 路径（runs/cke/final_library.json）")
    ap.add_argument("--outdir", default=None, help="自定义输出目录（默认 runs/<route>_<dataset>_<model>_<ts>）")
    args = ap.parse_args()

    cfg = get_config()
    model = cfg.model_main if args.model == "main" else cfg.model_debug
    scale_key = "koniq" if args.dataset.startswith("koniq") else "spaq"
    scale = cfg.scales[scale_key]

    images = load_images(cfg, args.dataset, limit=args.limit, seed=args.seed)
    print(f"[run] route={args.route} dataset={args.dataset} n={len(images)} model={model} scale={scale}")

    client = VLMClient(cfg, model)
    t0 = time.time()

    if args.route in ("r1b", "r1r"):
        rows = asyncio.run(run_r1(client, images, scale_key, args.route == "r1r", scale))
    else:
        assets = load_router_assets(cfg)
        rules_by_skill = None
        if args.library and os.path.exists(args.library):
            from iqa_agent.experience import ExperienceLibrary
            rules_by_skill = ExperienceLibrary.load(args.library).as_dict_by_skill()
            print(f"[run] 注入经验库: {args.library} ({sum(len(v) for v in rules_by_skill.values())} 条)")
        rows = asyncio.run(run_r2(
            client, images, scale_key, scale,
            dynamic=args.route == "r25",
            sensitivity=assets["sensitivity"], fitted=assets["fitted_weights"],
            rules_by_skill=rules_by_skill,
        ))
    dt = time.time() - t0

    # R3 = R2.5 + 规则库：route 标签改写为 r3（消融矩阵区分）
    if args.library and args.route == "r25":
        for r in rows:
            r["route"] = "r3"

    out_dir = args.outdir or os.path.join(cfg.runs_dir, f"{args.route}_{args.dataset}_{args.model}_{time.strftime('%m%d_%H%M')}")
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = ["img_id", "dataset", "route", "level", "score", "reason", "parse_tier",
                  "issues", "active_skills", "skill_scores"]
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_fail = sum(1 for r in rows if r["score"] is None)
    scores = [r["score"] for r in rows if r["score"] is not None]
    meta = {
        "route": args.route, "dataset": args.dataset, "model": model, "scale": scale,
        "n_images": len(images), "parse_failures": n_fail,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": sum(scores) / len(scores) if scores else None,
        "wall_time_s": round(dt, 1),
        "ledger": client.ledger(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] {out_dir}")
    print(f"[stats] parse_fail={n_fail}/{len(rows)}  mean={meta['score_mean']:.3f}  range=[{meta['score_min']}, {meta['score_max']}]")
    print(f"[ledger] {client.ledger()}")


if __name__ == "__main__":
    main()
