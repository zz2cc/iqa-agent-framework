# -*- coding: utf-8 -*-
"""二轮 D2：patch 放大镜 pilot + 期望分 pilot（门控决定进不进 R5）。

合规（ADR-0003）：仅 Train 像素，零 MOS；SPAQ 1568px 训练协议。

子命令：
  python scripts/89_pilots.py patch    # SPAQ 梯子 50 源：全局 vs 全局+双裁剪（noise/jpeg 端点）
  python scripts/89_pilots.py expect   # 100+100 张：单发 vs 分布直出 vs 释义×3（对 BT 一致率+扎堆度）
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.prompts.skills import build_skill_prompt, build_r1_rich_prompt
from iqa_agent.scoring import parse_score


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


# ---------- patch pilot ----------

def two_crops(img: Image.Image) -> list[Image.Image]:
    """中心 50% + 最高 Laplacian 方差窗口 50%（网格搜索）。"""
    w, h = img.size
    cw, ch = w // 2, h // 2
    crops = [img.crop((w // 4, h // 4, w // 4 + cw, h // 4 + ch))]
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    best, best_v = (0, 0), -1
    for gy in range(0, h - ch + 1, ch // 2):
        for gx in range(0, w - cw + 1, cw // 2):
            win = gray[gy:gy + ch, gx:gx + cw]
            c = win[1:-1, 1:-1]
            lap = -4 * c + win[:-2, 1:-1] + win[2:, 1:-1] + win[1:-1, :-2] + win[1:-1, 2:]
            v = float(np.var(lap))
            if v > best_v:
                best, best_v = (gx, gy), v
    crops.append(img.crop((best[0], best[1], best[0] + cw, best[1] + ch)))
    return crops


async def cmd_patch(cfg):
    d2 = os.path.join(cfg.runs_dir, "ladder2")
    manifest = jload(os.path.join(d2, "manifest.json"))
    scores = jload(os.path.join(d2, "scores.json"))
    img_dir = os.path.join(d2, "images")
    srcs = sorted({m["src_idx"] for m in manifest if m["domain"] == "spaq"})[:50]
    items = []
    for s in srcs:
        for fam in ("orig", "noise", "jpeg"):
            lv = 0 if fam == "orig" else 2
            f = [m["file"] for m in manifest if m["domain"] == "spaq" and m["src_idx"] == s
                 and m["family"] == fam and m["level"] == lv][0]
            items.append({"src": s, "fam": fam, "file": f})
    out_dir = os.path.join(cfg.runs_dir, "patch_pilot")
    os.makedirs(out_dir, exist_ok=True)
    cpath = os.path.join(out_dir, "crop_scores.json")
    crop_scores = jload(cpath) if os.path.exists(cpath) else {}
    client = VLMClient(cfg, cfg.model_main)

    async def one(it, ci):
        img = Image.open(os.path.join(img_dir, it["file"])).convert("RGB")
        crop = two_crops(img)[ci]
        tmp = os.path.join(out_dir, f"tmp_{it['src']}_{it['fam']}_{ci}.jpg")
        crop.save(tmp, "JPEG", quality=95)
        text, _ = await client.score_image(tmp, build_skill_prompt("S-TECH", "spaq", rules=None), temperature=0.0)
        p = parse_score(text, cfg.scales["spaq"])
        os.remove(tmp)
        return it["file"], ci, (p["score"] if p else None)

    jobs = [one(it, ci) for it in items for ci in (0, 1) if f"{it['file']}#{ci}" not in crop_scores]
    print(f"[patch] 待评 {len(jobs)} 次（{len(items)} 图 × 2 裁剪）")
    rows = await gather_with_progress(jobs, every=200, label="patch-crops")
    for r in rows:
        if not isinstance(r, Exception) and r[2] is not None:
            crop_scores[f"{r[0]}#{r[1]}"] = r[2]
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(crop_scores, f)
    print(f"[patch] 完成；账本 {client.ledger()}")

    # 融合规则离线对照（全局分来自梯子 scores.json）
    def fused(it, rule):
        g = scores.get(it["file"], {}).get("S-TECH")
        c0, c1 = crop_scores.get(f"{it['file']}#0"), crop_scores.get(f"{it['file']}#1")
        if g is None or c0 is None or c1 is None:
            return None
        cm = (c0 + c1) / 2
        if rule == "global":
            return g
        if rule == "mean":
            return 0.5 * g + 0.5 * cm
        if rule == "min":
            return min(g, cm + 0.5)
        if rule == "trust_crop":
            return cm if cm < g - 0.5 else g
        raise ValueError(rule)

    rules = ["global", "mean", "min", "trust_crop"]
    print(f"{'rule':11} {'noise 端点':>10} {'jpeg 端点':>10} {'合计':>8}")
    for rule in rules:
        acc = {}
        for fam in ("noise", "jpeg"):
            ok = tot = 0
            for s in srcs:
                o = fused(next(i for i in items if i["src"] == s and i["fam"] == "orig"), rule)
                hv = fused(next(i for i in items if i["src"] == s and i["fam"] == fam), rule)
                if o is not None and hv is not None:
                    ok += o > hv
                    tot += 1
            acc[fam] = ok / tot if tot else 0.0
        tot_acc = float(np.mean(list(acc.values())))
        print(f"{rule:11} {acc['noise']:10.3f} {acc['jpeg']:10.3f} {tot_acc:8.3f}")
    print("[patch] 门控：最优融合须比 global +0.05 才进 R5")


# ---------- expect pilot ----------

ELICIT_PROMPT = """Rate the overall quality of this image on a {lo}-{hi} scale.
Instead of a single number, give your belief as probabilities over five bands:
1 (very poor), 2 (poor), 3 (fair), 4 (good), 5 (excellent; map linearly to the {lo}-{hi} scale).
Reply with JSON only: {{"p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0, "p5": 0.0}}
(Probabilities must sum to 1.)"""

PARA_PROMPTS = [
    None,  # 原 rich prompt（索引 0 用原版）
    "Focus on technical fidelity (sharpness, noise, exposure, artifacts). " \
    "Rate the overall quality of this image from {lo} to {hi}. Reply with JSON only: " \
    "{{\"level\": <1-5>, \"score\": <{lo}-{hi} number>, \"reason\": \"<one sentence>\"}}",
    "You are a strict image quality rater. Considering clarity, noise, color and compression, " \
    "give a quality score from {lo} to {hi}. Reply with JSON only: " \
    "{{\"level\": <1-5>, \"score\": <{lo}-{hi} number>, \"reason\": \"<one sentence>\"}}",
    "Judge this photo's visual quality as an experienced reviewer would, from {lo} to {hi}. " \
    "Reply with JSON only: {{\"level\": <1-5>, \"score\": <{lo}-{hi} number>, \"reason\": \"<one sentence>\"}}",
]


def band_centers(lo, hi):
    return [lo + (hi - lo) * (k + 0.5) / 5 for k in range(5)]


async def cmd_expect(cfg, dom, n):
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    ranking = jload(os.path.join(wd, f"ranking_{dom}.json"))
    proto = jload(os.path.join(wd, f"protocol_scores_{dom}.json"))
    avail = [r for r in ranking if proto.get(r["path"], {}).get("rich") is not None]
    step = max(1, len(avail) // n)
    picks = avail[::step][:n]  # 跨排行榜均匀抽样（避免只取高分段）
    lo, hi = cfg.scales["koniq" if dom == "koniq" else "spaq"]
    client = VLMClient(cfg, cfg.model_main)
    out_path = os.path.join(cfg.runs_dir, "expect_pilot", f"scores_{dom}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    got = jload(out_path) if os.path.exists(out_path) else {}

    async def elicit(r):
        prompt = ELICIT_PROMPT.format(lo=int(lo), hi=int(hi))
        text, _ = await client.score_image(r["path"], prompt, temperature=0.0)
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
            ps = [float(obj[f"p{k}"]) for k in range(1, 6)]
            s = sum(ps)
            if s <= 0:
                return r["path"], "elicit", None
            exp = sum(p * c for p, c in zip(ps, band_centers(lo, hi))) / s
            return r["path"], "elicit", exp
        except (json.JSONDecodeError, KeyError, ValueError):
            return r["path"], "elicit", None

    async def para(r, k):
        prompt = build_r1_rich_prompt(dom) if k == 0 else PARA_PROMPTS[k].format(lo=int(lo), hi=int(hi))
        text, _ = await client.score_image(r["path"], prompt, temperature=0.0)
        p = parse_score(text, (lo, hi))
        return r["path"], f"para{k}", (p["score"] if p else None)

    jobs = [elicit(r) for r in picks if got.get(r["path"], {}).get("elicit") is None]
    jobs += [para(r, k) for r in picks for k in (1, 2, 3) if got.get(r["path"], {}).get(f"para{k}") is None]
    print(f"[expect:{dom}] 待评 {len(jobs)} 次（{len(picks)} 图）")
    rows = await gather_with_progress(jobs, every=100, label=f"expect-{dom}")
    for r in rows:
        if not isinstance(r, Exception) and r[2] is not None:
            got.setdefault(r[0], {})[r[1]] = r[2]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(got, f)
    print(f"[expect:{dom}] 完成；账本 {client.ledger()}")

    # 对照：单发 rich（87 缓存） vs 分布直出期望 vs 释义×3 均值
    bt = np.array([r["bt"] for r in picks])
    methods = {"single_rich": [], "elicit": [], "para3_mean": []}
    for r in picks:
        g = got.get(r["path"], {})
        methods["single_rich"].append(proto[r["path"]]["rich"])
        methods["elicit"].append(g.get("elicit"))
        ps = [g.get(f"para{k}") for k in (0, 1, 2, 3)]
        methods["para3_mean"].append(float(np.mean([p for p in ps if p is not None])) if any(p is not None for p in ps) else None)
    print(f"{'method':12} {'对BT一致率':>10} {'唯一值数':>8}")
    from scipy.stats import kendalltau
    for name, vals in methods.items():
        idx = [i for i, v in enumerate(vals) if v is not None]
        v = np.array([vals[i] for i in idx])
        tau = float(kendalltau(v, bt[idx]).statistic)
        print(f"{name:12} {tau:10.4f} {len(set(np.round(v, 4))):8d}")
    print("[expect] 门控：期望类方法 tau 不低于单发，且唯一值数显著更多（治扎堆）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["patch", "expect"])
    ap.add_argument("--domain", default="koniq", choices=["koniq", "spaq"])
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    cfg = get_config()
    if args.cmd == "patch":
        asyncio.run(cmd_patch(cfg))
    else:
        asyncio.run(cmd_expect(cfg, args.domain, args.n))


if __name__ == "__main__":
    main()
