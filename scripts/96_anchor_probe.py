# -*- coding: utf-8 -*-
"""F-024 锚点归因探针(post-hoc 分支,不读 MOS)。

臂 R1-anchor = bare 输出契约(只回一个数) + 与 R1-rich 完全相同的 level→band 锚点表
(iqa_agent.prompts.skills._SCALE_BLOCK 原文),无专家人设/检查清单/JSON 程序。
目的:把零点偏置分解为「锚点自致」与「模型裸先验」两层,并观察手写展宽先验的影响。

合规:只产出预测侧分布统计(n/mean/std/unique/带占比);对照用 MOS 均值与 std 为既往三次
评测已读之常数(见 findings F-020/F-023),硬编码于 MOS_KNOWN,全程不调用 load_mos——
零次新读 MOS。结果写入 runs/posthoc/,不触碰 runs/final/ 冻结主表。

用法: python scripts/96_anchor_probe.py [--limit N] [--domains koniq,spaq]
"""
import argparse
import asyncio
import csv
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

# GBK 控制台防御:进度警告含 emoji,打印失败不应杀死批量任务
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images
from iqa_agent.scoring import parse_score

try:
    from iqa_agent.prompts.skills import _SCALE_BLOCK
except ImportError:  # prompts 非包时按文件加载
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "iqa_agent", "prompts", "skills.py")
    _spec = importlib.util.spec_from_file_location("skills_mod", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    _SCALE_BLOCK = _m._SCALE_BLOCK

# 既往已读 MOS 汇总常数;本探针不再读取任何 MOS。
MOS_KNOWN = {"koniq": {"mean": 3.159, "std": 0.554},
             "spaq": {"mean": 4.795, "std": 2.194}}

# 锚点分数带(与 _SCALE_BLOCK 一致),用于统计预测落在各带的比例
BANDS = {"koniq": [(1.0, 1.4), (1.5, 2.4), (2.5, 3.4), (3.5, 4.2), (4.3, 5.0)],
         "spaq": [(0.0, 2.4), (2.5, 4.4), (4.5, 6.4), (6.5, 8.4), (8.5, 10.0)]}


def build_prompt(ds):
    return ("Rate the overall quality of this image.\n"
            + _SCALE_BLOCK[ds]
            + "\nReply with only a single number.")


async def run_domain(cfg, client, ds, split, limit, out_root):
    lo, hi = cfg.scales[ds]
    images = {r.img_id: r.path for r in load_images(cfg, split)}
    ids = sorted(images)[: limit or None]
    prompt = build_prompt(ds)
    cache_path = os.path.join(out_root, f"r1anchor_{ds}_raw.json")
    scores = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            scores = json.load(f)

    async def one(img_id):
        text, _ = await client.score_image(images[img_id], prompt, temperature=0.0)
        p = parse_score(text, (lo, hi))
        return img_id, (p["score"] if p else None)

    for rnd in (1, 2):  # 第二轮仅重试解析失败者
        todo = [i for i in ids if scores.get(i) is None]
        if not todo:
            break
        print(f"[{ds}] 第{rnd}轮 待评 {len(todo)}")
        jobs = [one(i) for i in todo]
        for c0 in range(0, len(jobs), 1000):
            part = await gather_with_progress(jobs[c0: c0 + 1000], every=250,
                                              label=f"anchor[{ds}][{c0 // 1000}]")
            for r in part:
                if not isinstance(r, Exception) and r[1] is not None:
                    scores[r[0]] = float(r[1])
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(scores, f)

    vals = np.array([scores[i] for i in ids if scores.get(i) is not None], dtype=float)
    out_dir = os.path.join(out_root, f"r1anchor_{ds}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "dataset", "route", "score"])
        for i in ids:
            if scores.get(i) is not None:
                w.writerow([i, ds, "r1anchor", round(scores[i], 4)])

    km = MOS_KNOWN[ds]
    band_frac = [float(np.mean((vals >= a) & (vals <= b))) for a, b in BANDS[ds]]
    summary = {
        "arm": f"r1anchor_{ds}", "n": int(len(vals)),
        "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
        "min": round(float(vals.min()), 3),
        "p25": round(float(np.percentile(vals, 25)), 3),
        "median": round(float(np.median(vals)), 3),
        "p75": round(float(np.percentile(vals, 75)), 3),
        "max": round(float(vals.max()), 3),
        "unique": int(len(set(np.round(vals, 4)))),
        "bias_vs_known_mos_mean": round(float(vals.mean()) - km["mean"], 4),
        "std_ratio_vs_known_mos": round(float(vals.std()) / km["std"], 3),
        "band_fraction_bad2excellent": [round(x, 4) for x in band_frac],
        "known_mos_constants": km,
        "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:12],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


async def main_async(args):
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)
    out_root = os.path.join(cfg.runs_dir, "posthoc")
    os.makedirs(out_root, exist_ok=True)
    splits = {"koniq": "koniq_val", "spaq": "spaq_test"}
    for ds in args.domains.split(","):
        await run_domain(cfg, client, ds.strip(), splits[ds.strip()], args.limit, out_root)
    print(f"[done] 账本 {client.ledger()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--domains", type=str, default="koniq,spaq")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
