# -*- coding: utf-8 -*-
"""二轮 D2：补齐协议分数（bare / rich）——协议路由训练的前置（§4.3 评估规则的选择）。

合规（ADR-0003）：仅 Train 像素，零 MOS；SPAQ 用 1568px 训练协议（已声明）。

输出 runs/full_tournament/protocol_scores_{dom}.json：
  {path: {"bare": score, "rich": score}}
KonIQ 999 张（CKE 工作集）×2 协议 ≈ ¥2；SPAQ 400 张 ×2 协议（1568px）≈ ¥4。

用法： python scripts/87_score_protocols.py --domain koniq|spaq
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.prompts.skills import build_r1_bare_prompt, build_r1_rich_prompt
from iqa_agent.scoring import parse_score


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
    if args.domain == "koniq":
        rows = jload(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"))
        paths = [r["path"] for r in rows if r.get("score") is not None]
    else:
        nodes = jload(os.path.join(wd, "nodes_spaq.json"))
        paths = [n["path"] for n in nodes]
    out_path = os.path.join(wd, f"protocol_scores_{args.domain}.json")
    scores = jload(out_path) if os.path.exists(out_path) else {}
    scale = cfg.scales["koniq" if args.domain == "koniq" else "spaq"]
    prompts = {"bare": build_r1_bare_prompt(args.domain), "rich": build_r1_rich_prompt(args.domain)}
    client = VLMClient(cfg, cfg.model_main)

    async def one(path, proto):
        text, _ = await client.score_image(path, prompts[proto], temperature=0.0)
        p = parse_score(text, scale)
        return path, proto, (p["score"] if p else None)

    jobs = [one(p, proto) for p in paths for proto in ("bare", "rich")
            if scores.get(p, {}).get(proto) is None]
    print(f"[{args.domain}] 待评 {len(jobs)} 次（{len(paths)} 图 × 2 协议，缓存自动跳过）")
    rows = await gather_with_progress(jobs, every=300, label="protocol-score")
    n_fail = 0
    for r in rows:
        if isinstance(r, Exception) or r[2] is None:
            n_fail += 1
            continue
        path, proto, sc = r
        scores.setdefault(path, {})[proto] = sc
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f)
    print(f"[{args.domain}] 落盘 {len(scores)} 图，解析失败 {n_fail}；账本 {client.ledger()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["koniq", "spaq"])
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
