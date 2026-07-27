# -*- coding: utf-8 -*-
"""SPAQ 运维试跑（S6）：300 张原图直传，只验证管线运维指标。

红线（ADR-0002 D5）：本脚本绝不读取 MOS、绝不计算 SRCC/MAE。
只记录：token 成本、延迟、解析成功率、分数分布 sanity。

用法： python scripts/60_pilot_spaq.py --n 300 --model main
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iqa_agent.config import get_config
from iqa_agent.client import VLMClient
from iqa_agent.data import load_images
from iqa_agent.pipeline import run_r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="main", choices=["main", "debug"])
    args = ap.parse_args()

    cfg = get_config()
    model = cfg.model_main if args.model == "main" else cfg.model_debug
    images = load_images(cfg, "spaq_test", limit=args.n, seed=cfg.seed)
    print(f"[pilot] SPAQ 运维试跑: {len(images)} 张原图直传, model={model}")
    print("[pilot] ⚠️ 红线：本试跑不读 MOS、不算 SRCC——只看运维指标")

    client = VLMClient(cfg, model)
    t0 = time.time()
    rows = asyncio.run(run_r1(client, images, "spaq", rich=True, scale=cfg.scales["spaq"]))
    dt = time.time() - t0

    ledger = client.ledger()
    n_fail = sum(1 for r in rows if r["score"] is None)
    scores = [r["score"] for r in rows if r["score"] is not None]
    report = {
        "n": len(rows), "parse_failures": n_fail,
        "wall_time_s": round(dt, 1), "calls_per_s": round(len(rows) / dt, 2),
        "tokens_in_per_call": ledger["tokens_in"] / max(1, ledger["api_calls"]),
        "tokens_out_per_call": ledger["tokens_out"] / max(1, ledger["api_calls"]),
        "est_cost_usd_per_1k": round(
            (ledger["tokens_in"] / max(1, ledger["api_calls"]) * 1000) / 1e6 * 0.287
            + (ledger["tokens_out"] / max(1, ledger["api_calls"]) * 1000) / 1e6 * 1.147, 3),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "ledger": ledger,
    }
    out = os.path.join(cfg.runs_dir, f"pilot_spaq_{args.model}_{time.strftime('%m%d_%H%M')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
