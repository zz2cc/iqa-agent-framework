# -*- coding: utf-8 -*-
"""二轮终评：R4/R5 臂执行（预注册 docs/预注册-R4R5.md 冻结版 v1.0）。

- R4-KonIQ：一轮 R2 缓存 5 专家分 × 学习融合权重（×IQR 兜底）——零新 API；
- R4-SPAQ：缓存 bare/rich/2 技能分 × 软门控（特征在 1568px 副本，仅 Router 用）——零新 API；
- R5-SPAQ：R4 + patch 放大镜 + para3 释义投票（新调用 ≈5 次/图）。

产出：runs/final/{r4_koniq,r4_spaq,r5_spaq}/scores.csv（格式同一轮，供 50_eval.py）。

用法：
  python scripts/90_run_r45.py --arm r4            # 双域 R4（零 API）
  python scripts/90_run_r45.py --arm r5 [--limit]  # SPAQ R5（新 API，~¥30）
"""
import argparse
import asyncio
import csv
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.data import load_images
from iqa_agent.prompts.skills import build_skill_prompt
from iqa_agent.router import iqr_adjusted_weights
from iqa_agent.scoring import parse_score

# 复用 89 的 para 文本与裁剪实现（保证与 pilot 完全一致）
_spec = importlib.util.spec_from_file_location(
    "pilots89", os.path.join(os.path.dirname(os.path.abspath(__file__)), "89_pilots.py"))
_p89 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p89)

SKILLS = ["S-TECH", "S-AESTH", "S-CONTENT", "S-NATURAL", "S-GLOBAL"]


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
            w.writerow({"img_id": r["img_id"], "dataset": dataset, "route": r["route"],
                        "level": "", "score": r["score"], "reason": r.get("reason", "")[:300],
                        "parse_tier": 1})
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"arm": name, "dataset": dataset, "n": len(rows), "prereg": "docs/预注册-R4R5.md"}, f)
    print(f"[{name}] {len(rows)} 行 → {out}")


# ---------- R4-KonIQ（零 API） ----------

def arm_r4_koniq(cfg):
    base = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r2_koniq", "scores.csv"))
    w = json.load(open(os.path.join(cfg.runs_dir, "router_v2", "weights_koniq.json")))["weights"]
    rows = []
    for img_id, r in base.items():
        try:
            ss = json.loads(r["skill_scores"])
        except (json.JSONDecodeError, TypeError):
            continue  # 一轮失败行（空 skill_scores）跳过，与一轮排除口径一致
        if any(ss.get(sk) is None for sk in SKILLS):
            continue
        adj = iqr_adjusted_weights(ss)
        num = sum(ss[sk] * w.get(sk, 0.0) * adj[sk] for sk in SKILLS)
        den = sum(w.get(sk, 0.0) * adj[sk] for sk in SKILLS)
        final = num / den if den > 0 else float(np.mean(list(ss.values())))
        top = sorted(SKILLS, key=lambda sk: -w.get(sk, 0) * adj[sk])[:3]
        reason = f"学习权重融合({','.join(f'{sk}×{w[sk]:.2f}' for sk in top)}) | " + r["reason"][:200]
        rows.append({"img_id": img_id, "route": "r4", "score": round(final, 4), "reason": reason})
    write_arm(cfg, "r4_koniq", "koniq", rows)


# ---------- R4-SPAQ（零 API，软门控） ----------

def opencv_features(img):
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


def load_spaq_gate(cfg):
    g = json.load(open(os.path.join(cfg.runs_dir, "router_v2", "route_spaq.json")))
    return g


def gate_weights(g, feat):
    raw = np.array([feat[k] for k in g["feat_keys"]], dtype=float)
    for j, k in enumerate(g["feat_keys"]):
        if k in ("lap_var", "noise", "colorful"):
            raw[j] = np.log(max(raw[j], 1e-6))
    z = (raw - np.array(g["mu"])) / np.array(g["sd"])
    W = np.array(g["W"])
    zz = W @ z
    zz = zz - zz.max()
    e = np.exp(zz)
    return e / e.sum()


def spaq_base_rows(cfg):
    r1b = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r1b_spaq", "scores.csv"))
    r1r = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r1r_spaq", "scores.csv"))
    r2 = read_scores_csv(os.path.join(cfg.runs_dir, "final", "r2_spaq", "scores.csv"))
    return r1b, r1r, r2


def arm_r4_spaq(cfg):
    from PIL import Image
    r1b, r1r, r2 = spaq_base_rows(cfg)
    g = load_spaq_gate(cfg)
    images = {r.img_id: r.path for r in load_images(cfg, "spaq_test")}
    rows = []
    for img_id, rb in r1b.items():
        rr = r1r.get(img_id, {})
        r2r = r2.get(img_id, {})
        try:
            bare = float(rb["score"])
            rich = float(rr["score"])
            ss = json.loads(r2r.get("skill_scores") or "")
            multi = (ss["S-TECH"] + ss["S-GLOBAL"]) / 2
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        # 门特征：1568px 降采样副本（仅 Router 用，D5 评分协议不变）
        img = Image.open(images[img_id])
        img.thumbnail((1568, 1568), Image.BICUBIC)
        feat = opencv_features(img)
        gw = gate_weights(g, feat)
        final = float(gw[0] * bare + gw[1] * rich + gw[2] * multi)
        reason = (f"软门控(bare {gw[0]:.2f}/rich {gw[1]:.2f}/multi {gw[2]:.2f}) | "
                  f"{(rr.get('reason') or '')[:180]}")
        rows.append({"img_id": img_id, "route": "r4", "score": round(final, 4), "reason": reason})
    write_arm(cfg, "r4_spaq", "spaq", rows)


# ---------- R5-SPAQ（新 API：para×3 + crops×2） ----------

async def arm_r5_spaq(cfg, limit):
    from PIL import Image
    r1b, r1r, r2 = spaq_base_rows(cfg)
    g = load_spaq_gate(cfg)
    images = {r.img_id: r.path for r in load_images(cfg, "spaq_test")}
    ids = list(r1b.keys())[: limit or None]
    lo, hi = cfg.scales["spaq"]
    client = VLMClient(cfg, cfg.model_main)
    cache_path = os.path.join(cfg.runs_dir, "r5_spaq_components.json")
    comp = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    async def para(img_id, k):
        prompt = _p89.PARA_PROMPTS[k].format(lo=int(lo), hi=int(hi))
        text, _ = await client.score_image(images[img_id], prompt, temperature=0.0)
        p = parse_score(text, (lo, hi))
        return img_id, f"para{k}", (p["score"] if p else None)

    async def crop(img_id, ci):
        img = Image.open(images[img_id]).convert("RGB")
        c = _p89.two_crops(img)[ci]
        import io as _io
        buf = _io.BytesIO()
        c.save(buf, "JPEG", quality=95)
        tmp = os.path.join(cfg.runs_dir, f"tmp_r5_{img_id}_{ci}.jpg")
        with open(tmp, "wb") as f:
            f.write(buf.getvalue())
        text, _ = await client.score_image(tmp, build_skill_prompt("S-TECH", "spaq", rules=None), temperature=0.0)
        p = parse_score(text, (lo, hi))
        os.remove(tmp)
        return img_id, f"crop{ci}", (p["score"] if p else None)

    jobs = []
    for img_id in ids:
        c = comp.get(img_id, {})
        for k in (1, 2, 3):
            if c.get(f"para{k}") is None:
                jobs.append(para(img_id, k))
        for ci in (0, 1):
            if c.get(f"crop{ci}") is None:
                jobs.append(crop(img_id, ci))
    print(f"[r5] 待调用 {len(jobs)} 次（{len(ids)} 图 × para3+crop2）")
    rows_api = []
    for chunk_i in range(0, len(jobs), 500):
        chunk = jobs[chunk_i: chunk_i + 500]
        part = await gather_with_progress(chunk, every=100, label=f"r5-components[{chunk_i // 500}]")
        rows_api.extend(part)
        for r in part:  # 增量落盘：断电/进程被杀也不丢已完成的组件分（F-020 教训）
            if not isinstance(r, Exception) and r[2] is not None:
                comp.setdefault(r[0], {})[r[1]] = r[2]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(comp, f)
    print(f"[r5] 组件分完成；账本 {client.ledger()}")

    rows = []
    for img_id in ids:
        rb, rr, r2r = r1b[img_id], r1r.get(img_id, {}), r2.get(img_id, {})
        c = comp.get(img_id, {})
        try:
            bare = float(rb["score"])
            paras = [c.get(f"para{k}") for k in (1, 2, 3)]
            rich = float(np.mean(paras)) if all(p is not None for p in paras) else float(rr["score"])
            ss = json.loads(r2r.get("skill_scores") or "")
            crops = [c.get(f"crop{ci}") for ci in (0, 1)]
            stech = 0.5 * ss["S-TECH"] + 0.5 * float(np.mean(crops)) if all(x is not None for x in crops) else ss["S-TECH"]
            multi = (stech + ss["S-GLOBAL"]) / 2
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        img = Image.open(images[img_id])
        img.thumbnail((1568, 1568), Image.BICUBIC)
        gw = gate_weights(g, opencv_features(img))
        final = float(gw[0] * bare + gw[1] * rich + gw[2] * multi)
        reason = f"R5=软门控+patch+para3 | bare {gw[0]:.2f}/rich {gw[1]:.2f}/multi {gw[2]:.2f}"
        rows.append({"img_id": img_id, "route": "r5", "score": round(final, 4), "reason": reason})
    write_arm(cfg, "r5_spaq", "spaq", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["r4", "r5"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config()
    if args.arm == "r4":
        arm_r4_koniq(cfg)
        arm_r4_spaq(cfg)
    else:
        asyncio.run(arm_r5_spaq(cfg, args.limit))


if __name__ == "__main__":
    main()
