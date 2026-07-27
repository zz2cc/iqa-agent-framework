# -*- coding: utf-8 -*-
"""阶梯评测（S3 后半）：5 Skill × 全部阶梯图评分 → 单调性 + 敏感度矩阵。

用途（PLAN §6）：
  ① Skill 体检：任一族单调性 <80% → 先修 prompt
  ② 敏感度矩阵 → Router 分诊依据（runs/ladder/sensitivity.json）

用法：
  python scripts/20_ladder_eval.py --model debug --limit 20   # 快速验证
  python scripts/20_ladder_eval.py --model main                # 正式（13,000 次调用）
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
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.prompts.skills import SKILL_ORDER, build_skill_prompt
from iqa_agent.scoring import parse_score


async def run_eval(client, manifest, img_dir, scale, src_limit=None):
    items = [m for m in manifest if src_limit is None or m["src_idx"] < src_limit]

    async def one(m, skill_id):
        prompt = build_skill_prompt(skill_id, "koniq")
        text, _ = await client.score_image(os.path.join(img_dir, m["file"]), prompt, temperature=0.0)
        p = parse_score(text, scale)
        return {
            "file": m["file"], "src_idx": m["src_idx"], "family": m["family"], "level": m["level"],
            "skill": skill_id, "score": p["score"] if p else None, "parse_ok": p is not None,
        }

    tasks = [one(m, s) for m in items for s in SKILL_ORDER]
    results = await gather_with_progress(tasks, every=500, label="ladder")
    n_api_fail = sum(1 for r in results if isinstance(r, Exception))
    if n_api_fail:
        print(f"  [ladder] API 失败 {n_api_fail} 条（重跑本脚本即可经缓存补齐）")
    return [r for r in results if not isinstance(r, Exception)]


def analyze(rows, out_dir):
    families = ["blur", "noise", "jpeg", "dark"]
    # 结构：by[(skill, src_idx)][family][level] = score（原图在 family="orig" level=0）
    by = {}
    for r in rows:
        by.setdefault((r["skill"], r["src_idx"]), {}).setdefault(r["family"], {})[r["level"]] = r["score"]

    mono = {}       # mono[skill][family] = [ok_pairs, total_pairs]
    sens = {}       # sens[skill][family] = [orig - L3, ...]
    for (skill, _src), fams in by.items():
        orig = fams.get("orig", {}).get(0)
        if orig is None:
            continue
        for family in families:
            lv = fams.get(family, {})
            seq = [orig, lv.get(1), lv.get(2), lv.get(3)]
            if any(v is None for v in seq):
                continue
            ok = sum(1 for a, b in zip(seq, seq[1:]) if a > b)
            d = mono.setdefault(skill, {}).setdefault(family, [0, 0])
            d[0] += ok
            d[1] += 3
            sens.setdefault(skill, {}).setdefault(family, []).append(seq[0] - seq[3])

    mono_report = {sk: {f: round(ok / tot, 4) for f, (ok, tot) in fams.items()} for sk, fams in mono.items()}
    sens_matrix = {sk: {f: round(sum(v) / len(v), 4) for f, v in fams.items()} for sk, fams in sens.items()}

    # 端点准确率：原图 > 最重档（这是"及格线"——最轻微的质量感也该做到）
    endpoint = {}
    for (skill, _src), fams in by.items():
        orig = fams.get("orig", {}).get(0)
        if orig is None:
            continue
        for family in families:
            sev = fams.get(family, {}).get(3)
            if sev is None:
                continue
            d = endpoint.setdefault(skill, {}).setdefault(family, [0, 0])
            d[0] += 1 if orig > sev else 0
            d[1] += 1
    endpoint_report = {sk: {f: round(ok / tot, 4) for f, (ok, tot) in fams.items()}
                       for sk, fams in endpoint.items()}
    with open(os.path.join(out_dir, "endpoint_accuracy.json"), "w") as f:
        json.dump(endpoint_report, f, indent=1)

    with open(os.path.join(out_dir, "monotonicity.json"), "w") as f:
        json.dump(mono_report, f, indent=1)
    with open(os.path.join(out_dir, "sensitivity.json"), "w") as f:
        json.dump(sens_matrix, f, indent=1)

    print("\n===== 单调性准确率（>0.8 合格）=====")
    print(f"{'skill':<10}" + "".join(f"{f:>8}" for f in families))
    for sk in SKILL_ORDER:
        print(f"{sk:<10}" + "".join(f"{mono_report.get(sk, {}).get(f, 0):>8.2f}" for f in families))
    print("\n===== 敏感度矩阵（原图均分 − 重档均分）=====")
    print(f"{'skill':<10}" + "".join(f"{f:>8}" for f in families))
    for sk in SKILL_ORDER:
        print(f"{sk:<10}" + "".join(f"{sens_matrix.get(sk, {}).get(f, 0):>8.2f}" for f in families))

    # 体检告警（双标准：相邻单调性 ≥0.8 为优；端点准确率 ≈1.0 为及格）
    print("\n===== 体检 =====")
    print("端点准确率（原图 > 最重档，≈1.0 为及格）：")
    print(f"{'skill':<10}" + "".join(f"{f:>8}" for f in families))
    for sk in SKILL_ORDER:
        print(f"{sk:<10}" + "".join(f"{endpoint_report.get(sk, {}).get(f, 0):>8.2f}" for f in families))
    bad = [(sk, f, v) for sk in SKILL_ORDER for f in families
           if (v := mono_report.get(sk, {}).get(f, 0)) < 0.8]
    if bad:
        print(f"\n相邻单调性 <0.8 的格子共 {len(bad)} 个（详见 monotonicity.json）")
    else:
        print("  ✅ 全部 Skill × 失真族单调性 ≥ 0.8")
    return mono_report, sens_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="debug", choices=["main", "debug"])
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 个源图（快速验证）")
    args = ap.parse_args()

    cfg = get_config()
    model = cfg.model_main if args.model == "main" else cfg.model_debug
    scale = cfg.scales["koniq"]

    with open(os.path.join(cfg.ladder_dir, "manifest.json")) as f:
        manifest = json.load(f)
    img_dir = os.path.join(cfg.ladder_dir, "images")

    client = VLMClient(cfg, model)
    n_calls = len([m for m in manifest if args.limit is None or m["src_idx"] < args.limit]) * 5
    print(f"[ladder-eval] model={model} calls={n_calls}")
    t0 = time.time()
    rows = asyncio.run(run_eval(client, manifest, img_dir, scale, args.limit))
    dt = time.time() - t0

    out_dir = os.path.join(cfg.ladder_dir, f"eval_{args.model}_{time.strftime('%m%d_%H%M')}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "src_idx", "family", "level", "skill", "score", "parse_ok"])
        w.writeheader()
        w.writerows(rows)

    n_fail = sum(1 for r in rows if not r["parse_ok"])
    print(f"[done] {len(rows)} rows in {dt:.0f}s, parse_fail={n_fail}")
    print(f"[ledger] {client.ledger()}")
    analyze(rows, out_dir)
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
