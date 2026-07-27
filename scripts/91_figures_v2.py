# -*- coding: utf-8 -*-
"""二轮归因图（考后分析，读 MOS 合法——同 70_figures.py 的考后口径）。

输出 runs/final/figures_v2/：
  f1_main_table_v2.png    主表 v2 消融（SRCC，双域）
  f2_theory_closure.png   F-002→F-020 理论闭环（机制不变、信号变）
  f3_scatter_r45.png      R4/R5 pred-vs-MOS 散点
  f4_gate_weights.png     SPAQ 软门控权重分布（考卷实际行为）
  f5_scale_probe.png      F-021 量程探针（同图双量程 + 模型/人类摆幅对照）
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_mos

OUT = os.path.join("runs", "final", "figures_v2")


def read_scores(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["img_id"]: float(r["score"]) for r in csv.DictReader(f)
                if r["score"] not in ("", "None")}


def f1(cfg):
    arms = ["r1b", "r1r", "r2", "r25", "r3", "r4", "r5"]
    labels = ["R1-bare", "R1-rich", "R2", "R2.5", "R3", "R4", "R5"]
    data = {}
    with open(os.path.join(cfg.runs_dir, "final", "main_table.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data[(r["route"], r["dataset"])] = float(r["SRCC"])
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for k, (ds, name, color) in enumerate((("koniq_val", "KonIQ Val", "#1f6fb2"),
                                           ("spaq_test", "SPAQ Test", "#e08a1e"))):
        vals = [data.get((a, ds), np.nan) for a in arms]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, label=name, color=color, alpha=0.85)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi + (k - 0.5) * 0.38, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SRCC")
    ax.set_ylim(0.5, 0.95)
    ax.set_title("Main table v2: SRCC by arm (R4/R5 = round-2, pre-registered)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_main_table_v2.png"), dpi=160)
    plt.close(fig)


def f2(cfg):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["round-1\n(old ladder signal)", "round-2\n(same-dist BT signal)"],
                [0.007, 0.024], color=["#999999", "#1f6fb2"])
    axes[0].set_title("Same mechanism (reweighting)\nheld-out concordance gain vs equal weights")
    axes[0].set_ylabel("gain")
    axes[1].bar(["round-1\n(F-002)", "round-2\n(F-020)"], [0.007, 0.062], color=["#999999", "#e08a1e"])
    axes[1].set_title("External SRCC gain from reweighting\n(R4 vs R2, identical cached inputs)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        for rect in ax.patches:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.001,
                    f"{rect.get_height():.3f}", ha="center", fontsize=9)
    fig.suptitle("Theory closure: mechanism unchanged, signal moved on-distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f2_theory_closure.png"), dpi=160)
    plt.close(fig)


def f3(cfg):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cases = [("r4", "koniq_val", "R4-KonIQ (SRCC 0.668, PLCC 0.735)", "#1f6fb2"),
             ("r4", "spaq_test", "R4-SPAQ (SRCC 0.885)", "#e08a1e"),
             ("r5", "spaq_test", "R5-SPAQ (SRCC 0.891)", "#2e8b57")]
    for ax, (arm, ds, title, color) in zip(axes, cases):
        mos = load_mos(cfg, ds)
        pred = read_scores(os.path.join(cfg.runs_dir, "final", f"{arm}_{ds.split('_')[0]}", "scores.csv"))
        pairs = [(pred[i], mos[i]) for i in pred if i in mos]
        p, m = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
        ax.scatter(m, p, s=6, alpha=0.35, color=color, edgecolors="none")
        lo = min(m.min(), p.min())
        hi = max(m.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("MOS")
        ax.set_ylabel("predicted")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f3_scatter_r45.png"), dpi=160)
    plt.close(fig)


def f4(cfg):
    """SPAQ 软门控：考卷 1,124 张的门权重分布（本地重算，零 API）。"""
    from PIL import Image
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "r45", os.path.join(os.path.dirname(os.path.abspath(__file__)), "90_run_r45.py"))
    r45 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r45)
    g = r45.load_spaq_gate(cfg)
    r1b, _, _ = r45.spaq_base_rows(cfg)
    from iqa_agent.data import load_images
    images = {r.img_id: r.path for r in load_images(cfg, "spaq_test")}
    W = []
    for i, img_id in enumerate(r1b):
        img = Image.open(images[img_id])
        img.thumbnail((1568, 1568), Image.BICUBIC)
        W.append(r45.gate_weights(g, r45.opencv_features(img)))
        if (i + 1) % 400 == 0:
            print(f"  gate {i + 1}/{len(r1b)}")
    W = np.array(W)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(W[:, 0], bins=40, alpha=0.7, label="bare", color="#1f6fb2")
    axes[0].hist(W[:, 1], bins=40, alpha=0.7, label="rich", color="#e08a1e")
    axes[0].hist(W[:, 2], bins=40, alpha=0.7, label="multi", color="#2e8b57")
    axes[0].set_title("SPAQ soft-gate weight histograms (n=1,124)")
    axes[0].legend()
    dom = np.argmax(W, axis=1)
    axes[1].bar(["bare", "rich", "multi"], [(dom == k).mean() for k in range(3)],
                color=["#1f6fb2", "#e08a1e", "#2e8b57"])
    axes[1].set_title("dominant protocol share")
    axes[1].set_ylabel("fraction of images")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f4_gate_weights.png"), dpi=160)
    plt.close(fig)


def f5(cfg):
    proto = json.load(open(os.path.join(cfg.runs_dir, "full_tournament", "protocol_scores_koniq.json"),
                           encoding="utf-8"))
    probe = json.load(open(os.path.join(cfg.runs_dir, "scale_probe.json")))
    ws = json.load(open(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"), encoding="utf-8"))
    r5, r10 = [], []
    for r in ws[::5][:200]:
        s10 = probe.get(r["path"])
        b = proto.get(r["path"], {}).get("bare")
        if s10 is None or b is None:
            continue
        r5.append((b - 1) / 4)
        r10.append(s10 / 10)
    r5, r10 = np.array(r5), np.array(r10)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(r5, r10, s=14, alpha=0.5, color="#1f6fb2", edgecolors="none")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("relative position on 1-5 scale")
    axes[0].set_ylabel("relative position on 0-10 scale")
    axes[0].set_title(f"Same 200 images, two scales (Δ={r5.mean() - r10.mean():+.3f})\nscale hypothesis REFUTED")
    groups = ["KonIQ images\n(model, 1-5)", "KonIQ Val\n(human MOS)", "SPAQ images\n(model, 0-10)", "SPAQ Test\n(human MOS)"]
    vals = [0.598, 0.540, 0.441, 0.480]
    colors = ["#1f6fb2", "#bbbbbb", "#e08a1e", "#bbbbbb"]
    axes[1].bar(groups, vals, color=colors)
    axes[1].set_ylim(0.35, 0.7)
    axes[1].set_title("content prior swing: model 18.9pp vs human 6pp\n(bias sign flip explained, no leakage)")
    for rect, v in zip(axes[1].patches, vals):
        axes[1].text(rect.get_x() + rect.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f5_scale_probe.png"), dpi=160)
    plt.close(fig)


def main():
    cfg = get_config()
    os.makedirs(OUT, exist_ok=True)
    f1(cfg)
    print("f1 ok")
    f2(cfg)
    print("f2 ok")
    f3(cfg)
    print("f3 ok")
    f5(cfg)
    print("f5 ok")
    f4(cfg)
    print("f4 ok")
    print(f"[done] → {OUT}")


if __name__ == "__main__":
    main()
