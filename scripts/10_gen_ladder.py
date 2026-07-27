# -*- coding: utf-8 -*-
"""生成合成失真阶梯（S3，纯本地，无 API）。

源图：KonIQ train 随机 200 张（seed=42，只用像素）。
四族失真 × 三档 + 原图 = 每源 13 张，共 2600 张。
排序真理：同一源同一族内 原图 > 轻 > 中 > 重（定义使然，零人工标注）。

输出：
  runs/ladder/images/{idx:03d}_{family}_{level}.jpg   （level: 0=原图, 1=轻, 2=中, 3=重）
  runs/ladder/manifest.json                            （全部元信息，供评测用）
"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_images

FAMILIES = {
    "blur":  [1.0, 2.0, 4.0],      # 高斯模糊 sigma
    "noise": [10.0, 25.0, 50.0],   # 高斯噪声 sigma (0-255)
    "jpeg":  [70, 40, 10],         # JPEG 质量
    "dark":  [0.8, 0.6, 0.4],      # 亮度倍率
}


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
    if family == "dark":
        return ImageEnhance.Brightness(img).enhance(param)
    raise ValueError(family)


def main():
    cfg = get_config()
    out_dir = os.path.join(cfg.ladder_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    sources = load_images(cfg, "koniq_train", limit=cfg.ladder_n_sources, seed=cfg.ladder_seed)
    print(f"[ladder] {len(sources)} 源图 × 13 版本")
    src_ids = [s.img_id for s in sources]
    with open(os.path.join(cfg.ladder_dir, "source_ids.json"), "w") as f:
        json.dump(src_ids, f)  # 供 CKE 工作集 exclude 用

    rng = np.random.default_rng(cfg.seed)
    manifest = []
    for idx, ref in enumerate(sources):
        img = Image.open(ref.path).convert("RGB")
        orig_name = f"{idx:03d}_orig_0.jpg"
        img.save(os.path.join(out_dir, orig_name), "JPEG", quality=95)
        manifest.append({"file": orig_name, "src_idx": idx, "family": "orig", "level": 0, "param": None})
        for family, params in FAMILIES.items():
            for lv, p in enumerate(params, start=1):
                out = apply_distortion(img, family, p, rng)
                name = f"{idx:03d}_{family}_{lv}.jpg"
                out.save(os.path.join(out_dir, name), "JPEG", quality=95)
                manifest.append({"file": name, "src_idx": idx, "family": family, "level": lv, "param": p})
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(sources)}")

    with open(os.path.join(cfg.ladder_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[done] {len(manifest)} 张 → {out_dir}")


if __name__ == "__main__":
    main()
