# -*- coding: utf-8 -*-
"""R6-offanchor 探针（post-hoc 分支，不碰冻结主表）。

消融目标：仅砍掉 _SCALE_BLOCK（5行数字带映射 + "Use the FULL range" 指令），
其余全部保留（人设/维度/检查清单/文字等级/5步程序/JSON 输出契约）。
bare 释义投票不动（本来就没锚点）。

KonIQ：3 专家 × 2014 重评 → 0.6·dynamic_fusion + 0.4·bare_vote
SPAQ：S-TECH/S-GLOBAL 重评（multi 槽）；rich 槽复用 R5 para1-3（PARA_PROMPTS
不含锚点表，已是轻锚点版）；bare 投票+crop 复用。软门控权重不变。

合规模态：MOS 用于离线性能评测（§4.5），第 4 次读取（考后诊断臂）；
产物入 runs/posthoc/，不触碰 runs/final/。

用法： python scripts/97_r6_offanchor.py [--domains koniq,spaq] [--limit N]
"""
import argparse
import asyncio
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images, load_mos
from iqa_agent.router import gate_weights, load_spaq_gate
from iqa_agent.prompts.skills import SKILLS, SKILL_ORDER
from iqa_agent.scoring import parse_score
from iqa_agent.metrics import compute_metrics

# ── 原始 prompt 构件（从 skills.py 复刻，保持字面一致） ──

_PROCEDURE = """\
[Assessment procedure — follow strictly]
1. Scan the whole image for a first impression.
2. Inspect each aspect in the checklist below, one by one.
3. Identify the DOMINANT quality issue(s) for your dimension.
4. Judge their severity objectively.
5. Map your judgment to a level, then give a precise score within that level's band."""

_OUTPUT_CONTRACT = """\
[Output format — strict]
Reply with JSON only, no other text:
{"level": <int 1-5>, "score": <float>, "reason": "<= 25 words, key evidence only"}"""

_SCALE_BLOCK = {
    "koniq": """\
[Score scale]
The final score is a float in [1, 5]. Level-to-band guide:
  5 Excellent -> 4.3-5.0 | 4 Good -> 3.5-4.2 | 3 Fair -> 2.5-3.4 | 2 Poor -> 1.5-2.4 | 1 Bad -> 1.0-1.4
Use the FULL range; do not cluster scores near the middle.""",
    "spaq": """\
[Score scale]
The final score is a float in [0, 10]. Level-to-band guide:
  5 Excellent -> 8.5-10 | 4 Good -> 6.5-8.4 | 3 Fair -> 4.5-6.4 | 2 Poor -> 2.5-4.4 | 1 Bad -> 0-2.4
Use the FULL range; do not cluster scores near the middle.""",
}

POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
ALPHA = 0.6

# ── 去锚点专家 prompt（仅砍 _SCALE_BLOCK，其余全部保留） ──

def build_offanchor_skill_prompt(skill_id: str, scale_key: str) -> str:
    """与 build_skill_prompt 完全一致，唯缺 _SCALE_BLOCK 段。"""
    s = SKILLS[skill_id]
    parts = [
        f"You are an expert image quality assessor specialized as a {s['name']}.",
        f"[Your dimension]\n{s['dimension']}",
        "[Checklist — inspect each aspect]\n" + "\n".join(
            f"- {name}: {desc}" for name, desc in s["checklist"]
        ),
        "[Quality levels for YOUR dimension]\n" + "\n".join(
            f"  {i + 1} = {txt}" for i, txt in enumerate(reversed(s["levels"]), start=0)
        ),
        _PROCEDURE,
        # 注意：_SCALE_BLOCK 在此有意省略——这是 offanchor 消融的唯一变量
        _OUTPUT_CONTRACT,
    ]
    return "\n\n".join(parts)


# ── 工具函数 ──

def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


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


def opencv_features(img):
    """纯 numpy/PIL 手工特征（与 iqa_agent.router.opencv_features 逐行一致）。"""
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    c = gray[1:-1, 1:-1]
    lap = -4 * c + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    lap_var = float(np.var(lap))
    box = (gray[:-2, :-2] + gray[:-2, 1:-1] + gray[:-2, 2:] + gray[1:-1, :-2] + c +
           gray[1:-1, 2:] + gray[2:, :-2] + gray[2:, 1:-1] + gray[2:, 2:]) / 9.0
    res = c - box
    gx = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2])
    thr = np.quantile(gx, 0.25)
    flat = res[gx <= thr]
    noise = float(1.4826 * np.median(np.abs(flat - np.median(flat)))) if flat.size else 0.0
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    h, w = gray.shape
    return {"lap_var": lap_var, "noise": noise, "colorful": colorful,
            "bright": float(gray.mean()), "logpix": float(np.log(h * w)), "aspect": float(w / h)}


# ══════════════════════════════════════════════════════════════════════
# KonIQ offanchor
# ══════════════════════════════════════════════════════════════════════

async def run_koniq_offanchor(cfg, client, args):
    lo, hi = cfg.scales["koniq"]
    fusion = jload(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"))
    images = {r.img_id: r.path for r in load_images(cfg, "koniq_val")}
    ids_all = sorted(images)

    # bare 投票缓存（R1b para0 + R6 para1-3，均无锚点，直接复用）
    r1b_path = os.path.join(cfg.runs_dir, "final", "r1b_koniq", "scores.csv")
    with open(r1b_path, encoding="utf-8-sig") as f:
        r1b = {r["img_id"]: float(r["score"]) for r in csv.DictReader(f)
               if r.get("score") not in ("", None)}
    para_cache_path = os.path.join(cfg.runs_dir, "r6_koniq_paras.json")
    para_cache = jload(para_cache_path) if os.path.exists(para_cache_path) else {}

    ids = ids_all[: args.limit or None]
    out_root = os.path.join(cfg.runs_dir, "posthoc")
    os.makedirs(out_root, exist_ok=True)

    # ── 三专家去锚点评分 ──
    expert_cache_path = os.path.join(out_root, "r6offanchor_koniq_experts.json")
    expert_scores = jload(expert_cache_path) if os.path.exists(expert_cache_path) else {}

    async def score_one_expert(img_id, sk):
        prompt = build_offanchor_skill_prompt(sk, "koniq")
        text, _ = await client.score_image(images[img_id], prompt, temperature=0.0)
        p = parse_score(text, (lo, hi))
        return img_id, sk, (p["score"] if p else None)

    for sk in POOL:
        jobs = [score_one_expert(i, sk) for i in ids
                if expert_scores.get(i, {}).get(sk) is None]
        if not jobs:
            print(f"[koniq] {sk} 全部缓存命中")
            continue
        print(f"[koniq] {sk} 待评 {len(jobs)}")
        for c0 in range(0, len(jobs), 1000):
            part = await gather_with_progress(jobs[c0: c0 + 1000], every=250,
                                              label=f"k-off-{sk[:4]}[{c0 // 1000}]")
            for r in part:
                if not isinstance(r, Exception) and r[2] is not None:
                    expert_scores.setdefault(r[0], {})[r[1]] = r[2]
            with open(expert_cache_path, "w", encoding="utf-8") as f:
                json.dump(expert_scores, f)

    # ── 融合 ──
    rows = []
    missing = 0
    for img_id in ids:
        bare_parts = [r1b.get(img_id)] + [para_cache.get(img_id, {}).get(str(k)) for k in (1, 2, 3)]
        if any(v is None for v in bare_parts):
            missing += 1; continue
        vote = float(np.mean(bare_parts))
        es = expert_scores.get(img_id, {})
        skills3 = [es.get(sk) for sk in POOL]
        if any(v is None for v in skills3):
            missing += 1; continue
        feat = opencv_features(Image.open(images[img_id]))
        fus, g = dynamic_fusion(fusion, skills3, feat)
        final = ALPHA * fus + (1 - ALPHA) * vote
        reason = (f"R6-offanchor=0.6·动态融合("
                  f"{','.join(f'{s}×{w:.2f}' for s, w in zip(POOL, g))})"
                  f"+0.4·bare投票({vote:.2f})")
        rows.append({"img_id": img_id, "score": round(final, 4), "reason": reason})

    print(f"[koniq] 有效 {len(rows)}/{len(ids)} (缺 {missing})")
    return rows, ids


# ══════════════════════════════════════════════════════════════════════
# SPAQ offanchor
# ══════════════════════════════════════════════════════════════════════

async def run_spaq_offanchor(cfg, client, args):
    lo, hi = cfg.scales["spaq"]
    images = {r.img_id: r.path for r in load_images(cfg, "spaq_test")}
    ids_all = sorted(images)
    ids = ids_all[: args.limit or None]
    out_root = os.path.join(cfg.runs_dir, "posthoc")

    # ── 加载 R6-SPAQ 全部组件缓存 ──
    # bare 投票（无锚点，复用）
    r1b_path = os.path.join(cfg.runs_dir, "final", "r1b_spaq", "scores.csv")
    with open(r1b_path, encoding="utf-8-sig") as f:
        r1b_s = {r["img_id"]: float(r["score"]) for r in csv.DictReader(f)
                 if r.get("score") not in ("", None)}
    para_s_cache = os.path.join(cfg.runs_dir, "r6_spaq_paras.json")
    para_s = jload(para_s_cache) if os.path.exists(para_s_cache) else {}

    # R5 组件分（rich para1-3 + crops —— rich 槽的 PARA_PROMPTS 不含锚点表，直接复用）
    comp_path = os.path.join(cfg.runs_dir, "r5_spaq_components.json")
    comp = jload(comp_path) if os.path.exists(comp_path) else {}

    # 软门控模型（从 iqa_agent.router 加载，缺文件时自动等权回退）

    # 读取 R2 SPAQ 缓存中的旧专家分（仅用于获取 id 列表，数值不用）
    r2_spaq_path = os.path.join(cfg.runs_dir, "final", "r2_spaq", "scores.csv")
    with open(r2_spaq_path, encoding="utf-8-sig") as f:
        r2_spaq = list(csv.DictReader(f))

    # ── S-TECH + S-GLOBAL 去锚点评分 ──
    expert_cache_path = os.path.join(out_root, "r6offanchor_spaq_experts.json")
    expert_scores = jload(expert_cache_path) if os.path.exists(expert_cache_path) else {}

    async def score_one_expert(img_id, sk):
        prompt = build_offanchor_skill_prompt(sk, "spaq")
        text, _ = await client.score_image(images[img_id], prompt, temperature=0.0)
        p = parse_score(text, (lo, hi))
        return img_id, sk, (p["score"] if p else None)

    for sk in ("S-TECH", "S-GLOBAL"):
        jobs = [score_one_expert(i, sk) for i in ids
                if expert_scores.get(i, {}).get(sk) is None]
        if not jobs:
            print(f"[spaq] {sk} 全部缓存命中")
            continue
        print(f"[spaq] {sk} 待评 {len(jobs)}")
        for c0 in range(0, len(jobs), 1000):
            part = await gather_with_progress(jobs[c0: c0 + 1000], every=250,
                                              label=f"s-off-{sk[:4]}[{c0 // 1000}]")
            for r in part:
                if not isinstance(r, Exception) and r[2] is not None:
                    expert_scores.setdefault(r[0], {})[r[1]] = r[2]
            with open(expert_cache_path, "w", encoding="utf-8") as f:
                json.dump(expert_scores, f)

    # ── 组装（与 R6-SPAQ 完全相同的结构，仅 multi 槽换去锚点分） ──
    rows = []
    missing = 0
    for img_id in ids:
        rb = r1b_s.get(img_id)
        c = comp.get(img_id, {})
        es = expert_scores.get(img_id, {})
        try:
            # bare 投票（无锚点，复用）
            bare_parts = [rb] + [para_s.get(img_id, {}).get(str(k)) for k in (1, 2, 3)]
            if any(v is None for v in bare_parts):
                missing += 1; continue
            bare = float(np.mean(bare_parts))

            # rich 槽：para1-3 复用 R5 缓存（PARA_PROMPTS 不含锚点表，已是轻锚点版）
            paras = [c.get(f"para{k}") for k in (1, 2, 3)]
            rich = float(np.mean(paras)) if all(p is not None for p in paras) else None
            if rich is None:
                missing += 1; continue

            # multi 槽：去锚点 S-TECH + S-GLOBAL（替换原 R2 锚点版）
            stech_off = es.get("S-TECH")
            sglobal_off = es.get("S-GLOBAL")
            if stech_off is None or sglobal_off is None:
                missing += 1; continue
            # crop 仍是 bare 调用，无锚点，复用
            crops = [c.get(f"crop{ci}") for ci in (0, 1)]
            if all(x is not None for x in crops):
                stech_off = 0.5 * stech_off + 0.5 * float(np.mean(crops))
            multi = (stech_off + sglobal_off) / 2
        except (TypeError, ValueError, KeyError):
            missing += 1; continue

        img = Image.open(images[img_id])
        img.thumbnail((1568, 1568), Image.BICUBIC)
        gw = gate_weights(load_spaq_gate(cfg), opencv_features(img))
        final = float(gw[0] * bare + gw[1] * rich + gw[2] * multi)
        reason = (f"R6-offanchor: 软门控 bare×{gw[0]:.2f} rich×{gw[1]:.2f} "
                  f"multi×{gw[2]:.2f} [multi=offanchor S-TECH+S-GLOBAL, rich=cached para3]")
        rows.append({"img_id": img_id, "score": round(final, 4), "reason": reason})

    print(f"[spaq] 有效 {len(rows)}/{len(ids)} (缺 {missing})")
    return rows, ids


# ══════════════════════════════════════════════════════════════════════
# 写入 + 评测
# ══════════════════════════════════════════════════════════════════════

def write_and_eval(cfg, rows, ids, ds, eval_ds):
    out_dir = os.path.join(cfg.runs_dir, "posthoc", f"r6offanchor_{ds}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "dataset", "route", "score",
                                          "reason", "parse_tier"])
        w.writeheader()
        for r in rows:
            w.writerow({"img_id": r["img_id"], "dataset": ds, "route": "r6offanchor",
                        "score": r["score"], "reason": r.get("reason", "")[:300],
                        "parse_tier": 1})

    pred = {r["img_id"]: r["score"] for r in rows}
    mos = load_mos(cfg, eval_ds)
    m = compute_metrics(pred, mos)
    vals = np.array(list(pred.values()))
    summary = {
        "arm": f"r6offanchor_{ds}",
        "n": m["n"], "SRCC": round(m["SRCC"], 4), "MAE": round(m["MAE"], 4),
        "PLCC": round(m["PLCC"], 4),
        "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
        "unique": int(len(set(np.round(vals, 4)))),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ══════════════════════════════════════════════════════════════════════

async def main_async(args):
    cfg = get_config()
    client = VLMClient(cfg, cfg.model_main)
    domains = [d.strip() for d in args.domains.split(",")]

    for ds in domains:
        if ds == "koniq":
            rows, ids = await run_koniq_offanchor(cfg, client, args)
            write_and_eval(cfg, rows, ids, "koniq", "koniq_val")
        elif ds == "spaq":
            rows, ids = await run_spaq_offanchor(cfg, client, args)
            write_and_eval(cfg, rows, ids, "spaq", "spaq_test")
    print(f"[done] 账本 {client.ledger()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--domains", type=str, default="koniq,spaq")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
