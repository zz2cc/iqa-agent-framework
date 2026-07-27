# -*- coding: utf-8 -*-
"""V3 终评：R6 臂执行（预注册 docs/预注册-R6.md 冻结版 v1.0）。

- R6-KonIQ = 0.6 × Router v3 逐图动态 3 专家融合（缓存 R2 专家分）+ 0.4 × bare 释义投票
  （原句缓存 + para1-3 新调用）；
- R6-SPAQ = R5 结构（软门控），bare 槽位换 bare 释义投票（其余组件分全部缓存）。

产出：runs/final/{r6_koniq,r6_spaq}/scores.csv（供 50_eval.py 第三次读 MOS）。

用法： python scripts/95_run_r6.py [--limit N]
"""
import argparse
import asyncio
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images
from iqa_agent.scoring import parse_score
from iqa_agent.prompts.skills import BARE_PARAS
from iqa_agent.router import opencv_features, spaq_base_rows, load_spaq_gate, gate_weights

POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
ALPHA = 0.6


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


def read_scores_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["img_id"]: r for r in csv.DictReader(f)}


def write_arm(cfg, name, dataset, rows):
    out = os.path.join(cfg.runs_dir, "final", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "level", "score",
                                          "reason", "parse_tier"])
        w.writeheader()
        for r in rows:
            w.writerow({"img_id": r["img_id"], "dataset": dataset, "route": "r6",
                        "level": "", "score": r["score"], "reason": r.get("reason", "")[:300],
                        "parse_tier": 1})
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"arm": name, "dataset": dataset, "n": len(rows), "prereg": "docs/预注册-R6.md"}, f)
    print(f"[{name}] {len(rows)} 行 → {out}")


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def dynamic_fusion(fusion, skills3, feat):
    f = np.array([feat[k] for k in fusion["feat_keys"][:-1]] + [0.0], dtype=float)
    f[6] = float(np.std(skills3))
    for j, k in enumerate(fusion["feat_keys"]):
        if k in ("lap_var", "noise", "colorful"):
            f[j] = np.log(max(f[j], 1e-6))
    sd = np.array([s if s > 1e-6 else 1.0 for s in fusion["sd"]])
    f = (f - np.array(fusion["mu"])) / sd
    g = softmax(np.array(fusion["W"]) @ f)
    return float(g @ np.array(skills3)), g


async def run_para_votes(cfg, client, images, ids, lo, hi, cache_path):
    """para1-3 释义投票（增量缓存）。"""
    votes = jload(cache_path) if os.path.exists(cache_path) else {}

    async def one(img_id, k):
        text, _ = await client.score_image(images[img_id], BARE_PARAS[k].format(lo=int(lo), hi=int(hi)),
                                           temperature=0.0)
        p = parse_score(text, (lo, hi))
        return img_id, k, (p["score"] if p else None)

    jobs = [one(i, k) for i in ids for k in (1, 2, 3) if votes.get(i, {}).get(str(k)) is None]
    print(f"  para 待评 {len(jobs)} 次（{len(ids)} 图 × 3 释义）")
    for chunk_i in range(0, len(jobs), 1000):
        chunk = jobs[chunk_i: chunk_i + 1000]
        part = await gather_with_progress(chunk, every=250, label=f"para[{chunk_i // 1000}]")
        for r in part:
            if not isinstance(r, Exception) and r[2] is not None:
                votes.setdefault(r[0], {})[str(r[1])] = r[2]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(votes, f)
    return votes


async def main_async(args):
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)
    lo_k, hi_k = cfg.scales["koniq"]

    # ---------- R6-KonIQ ----------
    base = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r2_koniq", "scores.csv"))
    r1b_k = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r1b_koniq", "scores.csv"))
    fusion = jload(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"))
    images_k = {r.img_id: r.path for r in load_images(cfg, "koniq_val")}
    ids_k = [i for i in base if i in r1b_k]
    ids_k = ids_k[: args.limit or None]
    votes = await run_para_votes(cfg, client, images_k, ids_k, lo_k, hi_k,
                                 os.path.join(cfg.runs_dir, "r6_koniq_paras.json"))
    rows = []
    for img_id in ids_k:
        r, rb = base.get(img_id), r1b_k.get(img_id)
        try:
            ss = json.loads(r["skill_scores"])
            skills3 = [ss[sk] for sk in POOL]
            bare_parts = [float(rb["score"])] + [votes.get(img_id, {}).get(str(k)) for k in (1, 2, 3)]
            if any(v is None for v in bare_parts):
                continue
            vote = float(np.mean(bare_parts))
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            continue
        feat = opencv_features(Image.open(images_k[img_id]))
        fus, g = dynamic_fusion(fusion, skills3, feat)
        final = ALPHA * fus + (1 - ALPHA) * vote
        reason = (f"R6=0.6·动态融合({','.join(f'{s}×{w:.2f}' for s, w in zip(POOL, g))})"
                  f"+0.4·bare投票({vote:.2f})")
        rows.append({"img_id": img_id, "score": round(final, 4), "reason": reason})
    write_arm(cfg, "r6_koniq", "koniq", rows)

    # ---------- R6-SPAQ ----------
    r1b, r1r, r2 = spaq_base_rows(cfg)
    g95 = load_spaq_gate(cfg)
    comp = jload(os.path.join(cfg.runs_dir, "r5_spaq_components.json"))
    images_s = {r.img_id: r.path for r in load_images(cfg, "spaq_test")}
    ids_s = list(r1b.keys())[: args.limit or None]
    lo_s, hi_s = cfg.scales["spaq"]
    votes_s = await run_para_votes(cfg, client, images_s, ids_s, lo_s, hi_s,
                                   os.path.join(cfg.runs_dir, "r6_spaq_paras.json"))
    rows = []
    for img_id in ids_s:
        rb, rr, r2r = r1b[img_id], r1r.get(img_id, {}), r2.get(img_id, {})
        c = comp.get(img_id, {})
        try:
            bare_parts = [float(rb["score"])] + [votes_s.get(img_id, {}).get(str(k)) for k in (1, 2, 3)]
            if any(v is None for v in bare_parts):
                continue
            bare = float(np.mean(bare_parts))
            paras = [c.get(f"para{k}") for k in (1, 2, 3)]
            rich = float(np.mean(paras)) if all(p is not None for p in paras) else float(rr["score"])
            ss = json.loads(r2r.get("skill_scores") or "")
            crops = [c.get(f"crop{ci}") for ci in (0, 1)]
            stech = 0.5 * ss["S-TECH"] + 0.5 * float(np.mean(crops)) if all(x is not None for x in crops) else ss["S-TECH"]
            multi = (stech + ss["S-GLOBAL"]) / 2
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        img = Image.open(images_s[img_id])
        img.thumbnail((1568, 1568), Image.BICUBIC)
        gw = gate_weights(g95, opencv_features(img))
        final = float(gw[0] * bare + gw[1] * rich + gw[2] * multi)
        reason = f"R6=R5结构+bare投票 | bare {gw[0]:.2f}/rich {gw[1]:.2f}/multi {gw[2]:.2f}"
        rows.append({"img_id": img_id, "score": round(final, 4), "reason": reason})
    write_arm(cfg, "r6_spaq", "spaq", rows)
    print(f"[done] 账本 {client.ledger()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
