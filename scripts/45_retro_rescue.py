# -*- coding: utf-8 -*-
"""Round 2 候选规则的补救重测（用户质询发现的漏洞：R2 候选被旧代码整批拒，未做逐条挽救）。

做法：精确重建 Round 2 状态（同锚点、同锦标赛 seed → 全缓存），
对其 5 条候选执行逐条阶梯门控 + 存活者 B-C 门控。
若有过门者，并入 runs/cke/final_library.json；否则输出 final_library = round1 的 3 条。
"""
import asyncio
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iqa_agent.config import get_config
from iqa_agent.client import VLMClient
from iqa_agent.cke import (anchor_b_scores, bc_disagreement, fit_bt, ladder_monotonicity,
                           run_tournament, tagged_skills_of)
from iqa_agent.experience import ExperienceLibrary
from iqa_agent.pipeline import run_r2
from iqa_agent.prompts.skills import SKILL_ORDER


async def main():
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)
    scale_key, scale = "koniq", cfg.scales["koniq"]

    log2 = json.load(open(os.path.join(cfg.runs_dir, "cke", "round2", "round_log.json")))
    anchor_ids, candidates = log2["anchors"], log2["candidates"]
    print(f"[retro] Round2 候选 {len(candidates)} 条，锚点 {len(anchor_ids)} 个")
    anchors = [{"img_id": i, "path": os.path.join(cfg.koniq_img_dir, i), "dataset": "koniq"}
               for i in anchor_ids]

    # 重建 Round 2 的 C 分（同 seed=cfg.seed+2 → 竞标赛全缓存）
    wins = await run_tournament(client, anchors, seed=cfg.seed + 2)
    c_latent = fit_bt(wins)

    # 重建 pre 状态（round1 规则库）：B 分与阶梯单调性（全缓存）
    lib = ExperienceLibrary.load(os.path.join(cfg.runs_dir, "cke", "round1", "library.json"))
    rules_pre = lib.as_dict_by_skill()
    b_rows = await run_r2(client, anchors, scale_key, scale, dynamic=False, rules_by_skill=rules_pre)
    b_scores = np.array([r["score"] if r["score"] is not None else np.nan for r in b_rows])
    dis_pre = float(np.nanmean(bc_disagreement(b_scores, c_latent)))

    with open(os.path.join(cfg.ladder_dir, "manifest.json")) as f:
        manifest = json.load(f)
    gate_items = [m for m in manifest if m["src_idx"] < 50]
    img_dir = os.path.join(cfg.ladder_dir, "images")
    mono_pre = await ladder_monotonicity(client, gate_items, img_dir, SKILL_ORDER, rules_pre, scale)
    print(f"[retro] pre 状态: mono={mono_pre:.3f} B-C分歧={dis_pre:.3f}")

    # 逐条阶梯门控
    survivors = []
    for cand in candidates:
        lib_one = ExperienceLibrary.load(os.path.join(cfg.runs_dir, "cke", "round1", "library.json"))
        lib_one.add(cand)
        sk = tagged_skills_of([cand])
        mono_one = await ladder_monotonicity(client, gate_items, img_dir, sk,
                                             lib_one.as_dict_by_skill(), scale)
        verdict = "KEEP" if mono_one >= mono_pre - 0.005 else "DROP"
        print(f"  {verdict} mono={mono_one:.3f} | {cand[:70]}")
        if verdict == "KEEP":
            survivors.append(cand)

    # 存活者合并过 B-C 门控
    accepted = []
    if survivors:
        lib_trial = ExperienceLibrary.load(os.path.join(cfg.runs_dir, "cke", "round1", "library.json"))
        for r in survivors:
            lib_trial.add(r)
        b_post = await anchor_b_scores(client, anchors, lib_trial.as_dict_by_skill(), scale_key, scale)
        dis_post = float(np.nanmean(bc_disagreement(b_post, c_latent)))
        print(f"[retro] B-C门控: {dis_pre:.3f}→{dis_post:.3f} {'PASS' if dis_post < dis_pre else 'FAIL'}")
        if dis_post < dis_pre:
            accepted = survivors[:3]

    # 冻结最终规则库
    final_lib = ExperienceLibrary.load(os.path.join(cfg.runs_dir, "cke", "round1", "library.json"))
    for r in accepted:
        final_lib.add(r)
    out_path = os.path.join(cfg.runs_dir, "cke", "final_library.json")
    final_lib.save(out_path)
    print(f"[retro] 挽救入库 {len(accepted)} 条；final_library 共 {len(final_lib)} 条 → {out_path}")
    print(f"[retro] {client.ledger()}")


if __name__ == "__main__":
    asyncio.run(main())
