# -*- coding: utf-8 -*-
"""数据层：图像清单加载与全项目唯一 MOS 入口。

合规设计（ADR-0001）：
- load_images() 只读取文件名/路径，绝不读取 MOS 列；
- load_mos() 是全项目唯一读取 MOS 的函数，仅 metrics.py / 50_eval.py / 70_figures.py 允许调用。
"""
import csv
import os
import random
from dataclasses import dataclass

from .config import Config


@dataclass
class ImageRef:
    img_id: str   # 仅用于本地 join，绝不进入 prompt
    path: str
    dataset: str  # koniq / spaq（仅本地标识）


def _read_ids(csv_path: str, id_col: str) -> list[str]:
    with open(csv_path, encoding="utf-8-sig") as f:
        return [r[id_col] for r in csv.DictReader(f)]


def load_images(cfg: Config, dataset: str, limit: int | None = None,
                seed: int | None = None, exclude: set[str] | None = None) -> list[ImageRef]:
    """加载图像清单。dataset ∈ {koniq_train, koniq_val, spaq_test}。
    limit 需配合 seed；exclude 用于保证工作集与阶梯源图不重叠。"""
    if dataset == "koniq_train":
        ids = _read_ids(cfg.koniq_train_csv, "img_id")
        img_dir, ds = cfg.koniq_img_dir, "koniq"
    elif dataset == "koniq_val":
        ids = _read_ids(cfg.koniq_val_csv, "img_id")
        img_dir, ds = cfg.koniq_img_dir, "koniq"
    elif dataset == "spaq_test":
        ids = _read_ids(cfg.spaq_test_csv, "image_id")
        img_dir, ds = cfg.spaq_img_dir, "spaq"
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    if exclude:
        ids = [i for i in ids if i not in exclude]
    if limit is not None:
        rng = random.Random(cfg.seed if seed is None else seed)
        ids = ids[:]  # 不改动原顺序
        rng.shuffle(ids)
        ids = ids[:limit]

    refs = []
    for i in ids:
        p = os.path.join(img_dir, i)
        if os.path.exists(p):
            refs.append(ImageRef(img_id=i, path=p, dataset=ds))
        else:
            raise FileNotFoundError(f"图像缺失: {p}")
    return refs


def load_mos(cfg: Config, dataset: str) -> dict[str, float]:
    """⚠️ 全项目唯一 MOS 读取入口。仅评测/考后分析脚本允许调用。"""
    if dataset == "koniq_val":
        with open(cfg.koniq_val_csv, encoding="utf-8-sig") as f:
            return {r["img_id"]: float(r["img_mos"]) for r in csv.DictReader(f)}
    if dataset == "spaq_test":
        with open(cfg.spaq_test_csv, encoding="utf-8-sig") as f:
            return {r["image_id"]: float(r["MOS"]) for r in csv.DictReader(f)}
    if dataset == "koniq_train":
        with open(cfg.koniq_train_csv, encoding="utf-8-sig") as f:
            return {r["img_id"]: float(r["img_mos"]) for r in csv.DictReader(f)}
    raise ValueError(f"unknown dataset: {dataset}")
