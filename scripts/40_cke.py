# -*- coding: utf-8 -*-
"""CKE 迭代入口（S5）。

用法： python scripts/40_cke.py --round 1 --model main
流程（PLAN §8）：工作集 B 路线 → 锚点 → 锦标赛 → BT → 分歧 → 裁判 → 双门控 → 经验库
全程零 MOS。
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iqa_agent.config import get_config
from iqa_agent.client import VLMClient
from iqa_agent.cke import (anchor_b_scores, bc_disagreement, fit_bt, ladder_monotonicity,
                           run_judge, run_tournament, select_anchors, tagged_skills_of, zscore)
from iqa_agent.data import load_images
from iqa_agent.experience import ExperienceLibrary
from iqa_agent.pipeline import run_r2


def round_dir(cfg, rnd):
    d = os.path.join(cfg.runs_dir, "cke", f"round{rnd}")
    os.makedirs(d, exist_ok=True)
    return d


async def main_async(args):
    cfg = get_config()
    model = cfg.model_main if args.model == "main" else cfg.model_debug
    scale_key, scale = "koniq", cfg.scales["koniq"]
    client = VLMClient(cfg, model)
    out = round_dir(cfg, args.round)

    # ---- 经验库（Round 0 空库起步；Round N 接上一轮冻结库）----
    lib = ExperienceLibrary()
    if args.round > 0:
        prev = os.path.join(cfg.runs_dir, "cke", f"round{args.round - 1}", "library.json")
        if os.path.exists(prev):
            lib = ExperienceLibrary.load(prev)
    print(f"[cke] round={args.round} model={model} 现有规则 {len(lib)} 条")

    # ---- Step 1: 工作集 B 路线（注入当前规则库）----
    with open(os.path.join(cfg.ladder_dir, "source_ids.json")) as f:
        ladder_src = set(json.load(f))
    workset = load_images(cfg, "koniq_train", limit=args.workset,
                          seed=cfg.workset_seed, exclude=ladder_src)
    print(f"[cke] Step1 工作集 B 路线: {len(workset)} 张 × 5 Skill")
    rules_by_skill = lib.as_dict_by_skill() if len(lib) else None
    b_rows = await run_r2(client, workset, scale_key, scale, dynamic=False,
                          rules_by_skill=rules_by_skill)
    b_rows = [dict(r, path=img.path) for r, img in zip(b_rows, workset)]

    # ---- Step 2-3: 锚点 + 锦标赛 ----
    anchors = select_anchors(b_rows, k=args.anchors)
    print(f"[cke] Step2-3 锚点 {len(anchors)} 个，锦标赛 {len(anchors)* (len(anchors)-1)//2} 场")
    wins = await run_tournament(client, anchors, seed=cfg.seed + args.round)

    # ---- Step 4-5: BT + 分歧 ----
    c_latent = fit_bt(wins)
    b_scores = np.array([r["score"] for r in anchors])
    dis = bc_disagreement(b_scores, c_latent)
    order = np.argsort(dis)
    g = args.groups
    agree_idx = order[:g].tolist()
    disagree_idx = order[-g:].tolist()
    b_z, c_z = zscore(b_scores), zscore(c_latent)
    print(f"[cke] Step4-5 分歧: mean={dis.mean():.3f} max={dis.max():.3f}")

    # ---- Step 6: 裁判（退火重排：R1=0.7 探索 → 每轮 -0.15 → 0.3 精炼；可用 --judge-temp 覆盖）----
    judge_temp = args.judge_temp if args.judge_temp is not None else max(0.3, 0.7 - 0.15 * (args.round - 1))
    print(f"[cke] Step6 裁判 (T={judge_temp:.2f})")
    raw, candidates = await run_judge(client, anchors, b_z, c_z,
                                      disagree_idx, agree_idx, judge_temp)
    print(f"[cke] 候选规则 {len(candidates)} 条:")
    for r in candidates:
        print(f"   {r[:100]}")

    # ---- Step 7: 双门控 ----
    with open(os.path.join(cfg.ladder_dir, "manifest.json")) as f:
        manifest = json.load(f)
    gate_items = [m for m in manifest if m["src_idx"] < args.gate_sources]
    img_dir = os.path.join(cfg.ladder_dir, "images")
    skills = tagged_skills_of(candidates)
    print(f"[cke] Step7 门控: 阶梯子集 {len(gate_items)} 图 × {len(skills)} Skill")

    mono_pre = await ladder_monotonicity(client, gate_items, img_dir, skills, rules_by_skill, scale)
    lib_trial = ExperienceLibrary.load(prev) if args.round > 0 and os.path.exists(prev) else ExperienceLibrary()
    for r in candidates:
        lib_trial.add(r)
    trial_by_skill = lib_trial.as_dict_by_skill()
    mono_post = await ladder_monotonicity(client, gate_items, img_dir, skills, trial_by_skill, scale)

    b_post = await anchor_b_scores(client, anchors, trial_by_skill, scale_key, scale)
    dis_pre_mean = float(dis.mean())
    dis_post = bc_disagreement(b_post, c_latent)
    dis_post_mean = float(np.nanmean(dis_post))

    gate_pass = (mono_post >= mono_pre - 0.005) and (dis_post_mean < dis_pre_mean)
    print(f"[cke] 门控: mono {mono_pre:.3f}→{mono_post:.3f} | B-C分歧 {dis_pre_mean:.3f}→{dis_post_mean:.3f} → {'PASS' if gate_pass else 'FAIL'}")

    # ---- Step 8: 经验库更新（整批 FAIL → 逐条挽救）----
    accepted = []
    if gate_pass:
        accepted = candidates[:3]
    else:
        print("[cke] 整批 FAIL，启动逐条挽救……")
        survivors = []
        for cand in candidates:
            lib_one = ExperienceLibrary.load(prev) if args.round > 0 and os.path.exists(prev) else ExperienceLibrary()
            lib_one.add(cand)
            sk = tagged_skills_of([cand])
            mono_one = await ladder_monotonicity(client, gate_items, img_dir, sk,
                                                 lib_one.as_dict_by_skill(), scale)
            verdict = "KEEP" if mono_one >= mono_pre - 0.005 else "DROP"
            print(f"   {verdict} mono={mono_one:.3f} | {cand[:70]}")
            if verdict == "KEEP":
                survivors.append(cand)
        if survivors:
            lib_trial2 = ExperienceLibrary.load(prev) if args.round > 0 and os.path.exists(prev) else ExperienceLibrary()
            for r in survivors:
                lib_trial2.add(r)
            b_post2 = await anchor_b_scores(client, anchors, lib_trial2.as_dict_by_skill(), scale_key, scale)
            dis_post2 = float(np.nanmean(bc_disagreement(b_post2, c_latent)))
            if dis_post2 < dis_pre_mean:
                accepted = survivors[:3]
                print(f"[cke] 挽救成功 {len(accepted)} 条（B-C分歧 {dis_pre_mean:.3f}→{dis_post2:.3f} ↓）")
            else:
                print(f"[cke] 挽救失败（B-C分歧 {dis_pre_mean:.3f}→{dis_post2:.3f} 未降）")
    for r in accepted:
        lib.add(r)
    if accepted:
        lib.keep_survivors()
    lib.save(os.path.join(out, "library.json"))

    # ---- 日志 ----
    log = {
        "round": args.round, "model": model, "judge_temp": judge_temp,
        "anchors": [a["img_id"] for a in anchors],
        "disagreement_mean": dis_pre_mean, "candidates": candidates,
        "gate": {"mono_pre": mono_pre, "mono_post": mono_post,
                 "dis_pre": dis_pre_mean, "dis_post": dis_post_mean, "pass": gate_pass},
        "accepted": accepted, "library_size": len(lib),
        "ledger": client.ledger(),
    }
    with open(os.path.join(out, "round_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "judge_raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"[cke] 轮次完成 → {out} | 规则库 {len(lib)} 条 | {client.ledger()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--model", default="main", choices=["main", "debug"])
    ap.add_argument("--total-rounds", type=int, default=3)
    ap.add_argument("--workset", type=int, default=1000)
    ap.add_argument("--anchors", type=int, default=50)
    ap.add_argument("--groups", type=int, default=15)
    ap.add_argument("--gate-sources", type=int, default=50)
    ap.add_argument("--judge-temp", type=float, default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))
