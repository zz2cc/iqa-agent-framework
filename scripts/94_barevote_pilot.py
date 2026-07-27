# -*- coding: utf-8 -*-
"""V3-D2：bare 释义投票 pilot（MAE 专项，V3-D）。

动机（F-019/F-020 校准分析）：bare 零点最准（MAE 全场最优）但只报整数（4-8 个唯一值，
量化粗）；给 bare 换 3 种释义各问一遍取平均——不动零点、只治量化。
门控（n=300，功效比二轮 n=100 翻倍）：投票版对 BT 的 Kendall τ 不低于单发 bare −0.01，
且唯一值数显著更多 → PASS 进 R6；否则回退单发（预注册写明）。

用法： python scripts/94_barevote_pilot.py --domain koniq|spaq [--n 300]
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.scoring import parse_score

BARE_PARAS = [
    "Rate the overall quality of this image on a scale from {lo} to {hi}. Reply with only a single number.",
    "On a scale from {lo} to {hi}, how would you rate the overall quality of this image? Reply with only a single number.",
    "Give a single overall quality score for this image, from {lo} (worst) to {hi} (best). Reply with only the number.",
    "As an image quality rater, assign one overall quality score from {lo} to {hi} to this image. Reply with only a single number.",
]


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


async def main_async(args):
    cfg = get_config()
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, "ranking_koniq_v3.json") if args.domain == "koniq" and os.path.exists(os.path.join(wd, "ranking_koniq_v3.json"))
                    else os.path.join(wd, f"ranking_{args.domain}.json"))
    proto = jload(os.path.join(wd, f"protocol_scores_{args.domain}.json"))
    lo, hi = cfg.scales["koniq" if args.domain == "koniq" else "spaq"]
    avail = [r for r in ranking if proto.get(r["path"], {}).get("bare") is not None]
    step = max(1, len(avail) // args.n)
    picks = avail[::step][: args.n]
    client = VLMClient(cfg, cfg.model_main)
    out_p = os.path.join(cfg.runs_dir, f"barevote_{args.domain}.json")
    got = jload(out_p) if os.path.exists(out_p) else {}

    async def one(r, k):
        text, _ = await client.score_image(r["path"], BARE_PARAS[k].format(lo=int(lo), hi=int(hi)), temperature=0.0)
        p = parse_score(text, (lo, hi))
        return r["path"], k, (p["score"] if p else None)

    jobs = [one(r, k) for r in picks for k in (1, 2, 3) if got.get(r["path"], {}).get(str(k)) is None]
    print(f"[{args.domain}] 待评 {len(jobs)} 次（{len(picks)} 图 × 3 释义，bare 原句缓存免费）")
    rows = await gather_with_progress(jobs, every=200, label="barevote")
    for r in rows:
        if not isinstance(r, Exception) and r[2] is not None:
            got.setdefault(r[0], {})[str(r[1])] = r[2]
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(got, f)
    print(f"完成；账本 {client.ledger()}")

    bt = np.array([r["bt"] for r in picks])
    single, vote = [], []
    for r in picks:
        single.append(proto[r["path"]]["bare"])
        vs = [proto[r["path"]]["bare"]] + [got.get(r["path"], {}).get(str(k)) for k in (1, 2, 3)]
        vs = [v for v in vs if v is not None]
        vote.append(float(np.mean(vs)) if vs else None)
    idx = [i for i, v in enumerate(vote) if v is not None]
    t_single = float(kendalltau(np.array(single)[idx], bt[idx]).statistic)
    t_vote = float(kendalltau(np.array(vote)[idx], bt[idx]).statistic)
    u_single = len(set(np.round(np.array(single)[idx], 3)))
    u_vote = len(set(np.round(np.array(vote)[idx], 3)))
    report = {"domain": args.domain, "n": len(idx),
              "tau_single": round(t_single, 4), "tau_vote": round(t_vote, 4),
              "unique_single": u_single, "unique_vote": u_vote,
              "gate_pass": bool(t_vote >= t_single - 0.01 and u_vote > u_single * 1.5)}
    with open(os.path.join(cfg.runs_dir, f"barevote_report_{args.domain}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"门控：τ 不降(≥单发-0.01)且唯一值 >1.5× → {'PASS（进 R6）' if report['gate_pass'] else 'FAIL（回退单发 bare）'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["koniq", "spaq"])
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
