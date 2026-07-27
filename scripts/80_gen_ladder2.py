# -*- coding: utf-8 -*-
"""大梯子 2.0（二轮 S1）：增量合成失真阶梯。纯本地生成零 API；评分单独子命令。

合规（ADR-0003）：仅用 KonIQ Train / SPAQ Train 像素，零 MOS；
文件名中的失真族/档位仅为本地 join 用，绝不进入任何线上 message。

设计（docs/二轮重设计计划-通俗版.md §3.1, v1.2 §11）：
- KonIQ：沿用第一轮 200 源（runs/ladder/images/*_orig_0.jpg 作基底），
  只新增 4 个新失真族 × 2 档；另增 100 个新源（避开阶梯旧源与 CKE 工作集）× 6 族 × 2 档；
  删除 dark 族（F-001 盲区 + 单调性不可靠），主打端点对（orig vs 重档）。
- SPAQ：150 源（需先按 --spaq-list 产出的名单解压到 runs/spaq_train_images/），
  训练信号统一缩到最长边 1568（降采样声明见 ADR-0003 §2），6 族 × 2 档。

子命令：
  python scripts/80_gen_ladder2.py spaq-list [--n 600]   # 生成 SPAQ Train 解压名单（只读 image_id 列）
  python scripts/80_gen_ladder2.py gen [--max-sources N] # 生成阶梯图像（--max-sources 供冒烟）
  python scripts/80_gen_ladder2.py score [--skills S-TECH,S-GLOBAL] [--all-skills-subset 0]
  python scripts/80_gen_ladder2.py report                # 端点准确率体检（门控 ≥0.80）+ 敏感度矩阵
"""
import argparse
import asyncio
import csv
import io
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_images

LADDER2 = "ladder2"
SPAQ_SRC_DIR = os.path.join("runs", "spaq_train_images")

# 新族参数（2 档：轻 / 重）；旧三族沿用第一轮档位的中/重
NEW_FAMILIES = {
    "down": [2, 4],        # 分辨率缩放因子（缩到 1/p 再放大回原尺寸）
    "band": [4, 3],        # posterize 位数
    "chroma": [0.5, 0.15], # 饱和度倍率
    "sharp": [150, 300],   # UnsharpMask percent
}
KONIQ_NEW_SRC_FAMILIES = {
    "blur": [2.0, 4.0], "noise": [25.0, 50.0], "jpeg": [40, 10],
    "down": [2, 4], "band": [4, 3], "chroma": [0.5, 0.15],
}
SPAQ_FAMILIES = {
    "blur": [1.5, 3.0], "noise": [15.0, 35.0], "jpeg": [50, 15],
    "down": [2, 4], "sharp": [150, 300], "band": [4, 3],
}
SPAQ_MAX_SIDE = 1568  # 训练信号降采样协议（ADR-0003 §2 声明）


def apply_distortion(img: Image.Image, family: str, param, rng: np.random.Generator) -> Image.Image:
    if family == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=param))
    if family == "noise":
        arr = np.asarray(img, dtype=np.float32)
        arr = np.clip(arr + rng.normal(0, param, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if family == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if family == "down":
        w, h = img.size
        small = img.resize((max(1, w // param), max(1, h // param)), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if family == "band":
        return ImageOps.posterize(img, int(param))
    if family == "chroma":
        return ImageEnhance.Color(img).enhance(param)
    if family == "sharp":
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(param), threshold=2))
    raise ValueError(family)


def emit(manifest, out_dir, img, src_idx, family, level, param, domain, prefix=""):
    name = f"{prefix}{src_idx:03d}_{family}_{level}.jpg"
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):  # 已存在则跳过写入（重跑只补新增段）
        img.save(path, "JPEG", quality=95)
    manifest.append({"file": name, "src_idx": src_idx, "family": family,
                     "level": level, "param": param, "domain": domain})


def cmd_spaq_list(cfg, n: int):
    """从 spaqTrain.csv 抽 n 个 image_id（只读 id 列，零 MOS），生成 7z 解压名单。"""
    ids = []
    with open(cfg.spaq_test_csv.replace("spaqTest.csv", "spaqTrain.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ids.append(r["image_id"])  # ⚠️ 只读 image_id 列（ADR-0003 §2）
    rng = np.random.default_rng(cfg.seed)
    pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
    out_dir = os.path.join(cfg.runs_dir, LADDER2)
    os.makedirs(out_dir, exist_ok=True)
    list_path = os.path.join(out_dir, "spaq_pick_list.txt")
    with open(list_path, "w") as f:
        for i in sorted(pick):
            f.write(f"TestImage\\{i}\n")  # zip 内目录名为 TestImage（实测）
    print(f"[spaq-list] {len(pick)} 个 Train id → {list_path}")
    print("解压命令（在 评测数据集 下执行，约数分钟）：")
    print(f'  7z x "SPAQ zip/SPAQ.zip" -i@"{os.path.relpath(list_path)}" -o"SPAQ" -y')
    print(f'解压后把图像移动到 {SPAQ_SRC_DIR}/ （扁平化，仅保留 .jpg）')


def cmd_gen(cfg, max_sources: int | None):
    out_dir = os.path.join(cfg.runs_dir, LADDER2, "images")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 1)
    manifest = []

    # ---- KonIQ 旧 200 源：基底 = 一轮 ladder 的 orig 图，只新增 4 族 × 2 档 ----
    old_base_dir = os.path.join(cfg.ladder_dir, "images")
    n_old = cfg.ladder_n_sources if max_sources is None else min(max_sources, cfg.ladder_n_sources)
    for idx in range(n_old):
        base_path = os.path.join(old_base_dir, f"{idx:03d}_orig_0.jpg")
        # 原图复制进 ladder2 并登记（report 需要 orig 分数作端点对照）
        dst_name = f"{idx:03d}_orig_0.jpg"
        dst = os.path.join(out_dir, dst_name)
        if not os.path.exists(dst):
            shutil.copy(base_path, dst)
        manifest.append({"file": dst_name, "src_idx": idx, "family": "orig",
                         "level": 0, "param": None, "domain": "koniq"})
        img = Image.open(base_path).convert("RGB")
        for family, params in NEW_FAMILIES.items():
            for lv, p in enumerate(params, start=1):
                out = apply_distortion(img, family, p, rng)
                emit(manifest, out_dir, out, idx, family, lv, p, "koniq")
        if (idx + 1) % 50 == 0:
            print(f"  [koniq 旧源新族] {idx + 1}/{n_old}")

    # ---- KonIQ 新 100 源（避开一轮阶梯源与 CKE 工作集）：6 族 × 2 档 + 原图 ----
    with open(os.path.join(cfg.ladder_dir, "source_ids.json")) as f:
        ladder_src = set(json.load(f))
    workset_ids = {r.img_id for r in load_images(cfg, "koniq_train", limit=cfg.workset_size,
                                                 seed=cfg.workset_seed, exclude=ladder_src)}
    n_new = 100 if max_sources is None else max_sources
    new_sources = load_images(cfg, "koniq_train", limit=n_new, seed=cfg.seed + 1000,
                              exclude=ladder_src | workset_ids)
    for j, ref in enumerate(new_sources):
        src_idx = cfg.ladder_n_sources + j  # 200..
        img = Image.open(ref.path).convert("RGB")
        emit(manifest, out_dir, img, src_idx, "orig", 0, None, "koniq")
        for family, params in KONIQ_NEW_SRC_FAMILIES.items():
            for lv, p in enumerate(params, start=1):
                out = apply_distortion(img, family, p, rng)
                emit(manifest, out_dir, out, src_idx, family, lv, p, "koniq")

    # ---- SPAQ 150 源（若已解压）：先统一缩到 1568，再 6 族 × 2 档 + 原图 ----
    spaq_dir = os.path.join(cfg.runs_dir, "spaq_train_images")
    if os.path.isdir(spaq_dir):
        spaq_files = sorted(f for f in os.listdir(spaq_dir) if f.lower().endswith(".jpg"))
        rng_p = np.random.default_rng(cfg.seed + 2000)
        pick = rng_p.choice(spaq_files, size=min(150 if max_sources is None else max_sources,
                                                 len(spaq_files)), replace=False)
        for k, fn in enumerate(sorted(pick)):
            src_idx = 300 + k
            img = Image.open(os.path.join(spaq_dir, fn)).convert("RGB")
            img.thumbnail((SPAQ_MAX_SIDE, SPAQ_MAX_SIDE), Image.BICUBIC)  # 降采样协议
            emit(manifest, out_dir, img, src_idx, "orig", 0, None, "spaq")
            for family, params in SPAQ_FAMILIES.items():
                for lv, p in enumerate(params, start=1):
                    out = apply_distortion(img, family, p, rng)
                    emit(manifest, out_dir, out, src_idx, family, lv, p, "spaq")
            if (k + 1) % 50 == 0:
                print(f"  [spaq] {k + 1}/{len(pick)}")
    else:
        print(f"[spaq] {spaq_dir} 不存在，跳过（先运行 spaq-list 子命令并解压）")

    mpath = os.path.join(cfg.runs_dir, LADDER2, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    n_k = sum(1 for m in manifest if m["domain"] == "koniq")
    print(f"[done] {len(manifest)} 张（koniq {n_k} / spaq {len(manifest) - n_k}）→ {out_dir}")


async def cmd_score(cfg, skills: list[str], all_skills_subset: int):
    from iqa_agent.client import VLMClient, gather_with_progress
    from iqa_agent.prompts.skills import SKILL_ORDER, build_skill_prompt
    from iqa_agent.scoring import parse_score

    mpath = os.path.join(cfg.runs_dir, LADDER2, "manifest.json")
    with open(mpath) as f:
        manifest = json.load(f)
    img_dir = os.path.join(cfg.runs_dir, LADDER2, "images")
    spath = os.path.join(cfg.runs_dir, LADDER2, "scores.json")
    scores = json.load(open(spath)) if os.path.exists(spath) else {}

    client = VLMClient(cfg, cfg.model_main)

    async def one(item, sk):
        scale = cfg.scales["koniq"] if item["domain"] == "koniq" else cfg.scales["spaq"]
        scale_key = "koniq" if item["domain"] == "koniq" else "spaq"
        prompt = build_skill_prompt(sk, scale_key, rules=None)
        text, _ = await client.score_image(os.path.join(img_dir, item["file"]), prompt, temperature=0.0)
        p = parse_score(text, scale)
        return item["file"], sk, (p["score"] if p else None)

    tasks = [one(it, sk) for it in manifest for sk in
             (SKILL_ORDER if (it["src_idx"] % 300) < all_skills_subset else skills)
             if sk not in scores.get(it["file"], {})]
    print(f"[score] 待评 {len(tasks)} 次（缓存命中自动跳过）")
    rows = await gather_with_progress(tasks, every=500, label="ladder2-score")
    n_fail = 0
    for r in rows:
        if isinstance(r, Exception) or r[2] is None:
            n_fail += 1
            continue
        fn, sk, sc = r
        scores.setdefault(fn, {})[sk] = sc
    with open(spath, "w") as f:
        json.dump(scores, f)
    print(f"[score] 完成，解析失败 {n_fail}；账本 {client.ledger()}")


def cmd_report(cfg):
    d = os.path.join(cfg.runs_dir, LADDER2)
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    scores = json.load(open(os.path.join(d, "scores.json")))
    by = {}
    for it in manifest:
        for sk, sc in scores.get(it["file"], {}).items():
            if sc is not None:
                by.setdefault((it["domain"], it["src_idx"], it["family"], sk), {})[it["level"]] = sc

    endpoint, sensitivity = {}, {}
    print(f"{'domain':6} {'skill':9} {'family':7} {'orig>L2':>8} {'orig>L1':>8} {'L1>L2':>6} {'gate':>5}")
    for (dom, fam, sk) in sorted({(d2, f2, s2) for (d2, _, f2, s2) in by}):
        if fam == "orig":
            continue
        e_ok = e_tot = o1_ok = o1_tot = a_ok = a_tot = 0
        diffs = []
        srcs = {s2 for (d3, s2, f3, s3) in by if d3 == dom and f3 == fam and s3 == sk}
        for s in srcs:
            orig = by.get((dom, s, "orig", sk), {}).get(0)
            l1 = by.get((dom, s, fam, sk), {}).get(1)
            l2 = by.get((dom, s, fam, sk), {}).get(2)
            if orig is None or l2 is None:
                continue
            e_tot += 1
            e_ok += orig > l2
            diffs.append(orig - l2)
            if l1 is not None:
                o1_tot += 1
                o1_ok += orig > l1
                a_tot += 1
                a_ok += l1 > l2
        if not e_tot:
            continue
        ep = e_ok / e_tot
        endpoint.setdefault(dom, {})[f"{sk}/{fam}"] = ep
        sensitivity.setdefault(sk, {})[fam] = round(float(np.mean(diffs)), 4)
        print(f"{dom:6} {sk:9} {fam:7} {ep:8.3f} {(o1_ok / o1_tot if o1_tot else 0):8.3f} "
              f"{(a_ok / a_tot if a_tot else 0):6.3f} {'PASS' if ep >= 0.80 else 'FAIL':>5}")
    with open(os.path.join(d, "endpoint_accuracy.json"), "w") as f:
        json.dump(endpoint, f, indent=1)
    with open(os.path.join(d, "sensitivity.json"), "w") as f:
        json.dump(sensitivity, f, indent=1)
    print(f"[report] → {d}/endpoint_accuracy.json, sensitivity.json（门控：端点 ≥0.80）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score", "report", "spaq-list"])
    ap.add_argument("--n", type=int, default=600, help="spaq-list 抽样数")
    ap.add_argument("--max-sources", type=int, default=None, help="冒烟用：每段最多处理多少源")
    ap.add_argument("--skills", default="S-TECH,S-GLOBAL")
    ap.add_argument("--all-skills-subset", type=int, default=0)
    args = ap.parse_args()
    cfg = get_config()
    if args.cmd == "spaq-list":
        cmd_spaq_list(cfg, args.n)
    elif args.cmd == "gen":
        cmd_gen(cfg, args.max_sources)
    elif args.cmd == "score":
        asyncio.run(cmd_score(cfg, args.skills.split(","), args.all_skills_subset))
    elif args.cmd == "report":
        cmd_report(cfg)


if __name__ == "__main__":
    main()
