# -*- coding: utf-8 -*-
"""验收图表生成（S8）。考后分析专用，允许读取 MOS（§4.5 离线误差分析）。

用法： python scripts/70_figures.py --final runs/final
产出： runs/final/figures/*.png
"""
import argparse
import csv
import glob
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROUTE_LABEL = {"r1b": "R1-bare", "r1r": "R1-rich", "r2": "R2", "r25": "R2.5", "r3": "R3"}


def fig_scatter(final_dir, cfg, fig_dir):
    """pred-vs-MOS 散点图：每路线 × 数据集一张。"""
    made = []
    for scores_path in glob.glob(os.path.join(final_dir, "*", "scores.csv")):
        run = os.path.basename(os.path.dirname(scores_path))
        with open(scores_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        ds = "koniq_val" if rows[0]["dataset"] == "koniq" else "spaq_test"
        mos = load_mos(cfg, ds)
        pairs = [(float(r["score"]), mos[r["img_id"]]) for r in rows
                 if r["score"] not in ("", "None", None) and r["img_id"] in mos]
        if not pairs:
            continue
        p, m = zip(*pairs)
        srcc = float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(m)))[0, 1])
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.scatter(m, p, s=4, alpha=0.25)
        ax.set_xlabel("MOS")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{ROUTE_LABEL.get(rows[0]['route'], run)} @ {ds}  (SRCC={srcc:.3f})")
        out = os.path.join(fig_dir, f"scatter_{run}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        made.append(out)
    return made


def fig_ablation(final_dir, fig_dir):
    """消融柱状图：SRCC 与 MAE 双子图。"""
    table_path = os.path.join(final_dir, "main_table.csv")
    if not os.path.exists(table_path):
        return []
    with open(table_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    for ax, metric, title in [(axes[0], "SRCC", "SRCC ↑"), (axes[1], "MAE", "MAE ↓")]:
        x = np.arange(len(datasets))
        routes = [r for r in ["r1b", "r1r", "r2", "r25", "r3"] if any(t["route"] == r for t in rows)]
        width = 0.8 / max(1, len(routes))
        for i, rt in enumerate(routes):
            vals = []
            for ds in datasets:
                hit = [t for t in rows if t["route"] == rt and t["dataset"] == ds]
                vals.append(float(hit[0][metric]) if hit and hit[0][metric] not in ("", "None") else np.nan)
            ax.bar(x + i * width, vals, width, label=ROUTE_LABEL.get(rt, rt))
        ax.set_xticks(x + width * len(routes) / 2)
        ax.set_xticklabels(datasets)
        ax.set_title(title)
        ax.legend(fontsize=8)
    out = os.path.join(fig_dir, "ablation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return [out]


def fig_sensitivity(cfg, fig_dir):
    """阶梯敏感度热图（5 Skill × 4 失真族）。"""
    evals = sorted(glob.glob(os.path.join(cfg.ladder_dir, "eval_main_*", "sensitivity.json"))
                   + glob.glob(os.path.join(cfg.ladder_dir, "eval_debug_*", "sensitivity.json")), reverse=True)
    if not evals:
        return []
    with open(evals[0], encoding="utf-8") as f:
        sens = json.load(f)
    skills = ["S-TECH", "S-AESTH", "S-CONTENT", "S-NATURAL", "S-GLOBAL"]
    fams = ["blur", "noise", "jpeg", "dark"]
    M = np.array([[sens.get(sk, {}).get(fm, 0) for fm in fams] for sk in skills])
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(fams)), fams)
    ax.set_yticks(range(len(skills)), skills)
    for i in range(len(skills)):
        for j in range(len(fams)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=9)
    ax.set_title("Ladder sensitivity (score drop: orig − severe)")
    fig.colorbar(im)
    out = os.path.join(fig_dir, "sensitivity_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return [out]


def fig_cke_evolution(cfg, fig_dir):
    """CKE 规则库演化：每轮门控指标与规则数。"""
    logs = sorted(glob.glob(os.path.join(cfg.runs_dir, "cke", "round*", "round_log.json")))
    if not logs:
        return []
    rounds, mono_pre, mono_post, dis_pre, dis_post, lib_size = [], [], [], [], [], []
    for p in logs:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("model") and "32b" not in d["model"]:
            continue  # 只画 32B 正式轮次（排除 8B 烟测 round0）
        rounds.append(d["round"])
        mono_pre.append(d["gate"]["mono_pre"])
        mono_post.append(d["gate"]["mono_post"])
        dis_pre.append(d["gate"]["dis_pre"])
        dis_post.append(d["gate"]["dis_post"])
        lib_size.append(d["library_size"])
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=120)
    ax1.plot(rounds, mono_pre, "o--", label="ladder mono (pre)")
    ax1.plot(rounds, mono_post, "o-", label="ladder mono (post)")
    ax1.plot(rounds, dis_pre, "s--", label="B-C disagreement (pre)")
    ax1.plot(rounds, dis_post, "s-", label="B-C disagreement (post)")
    ax1.set_xlabel("CKE round")
    ax1.legend(fontsize=8)
    ax2 = ax1.twinx()
    ax2.bar(rounds, lib_size, alpha=0.2, color="gray", label="library size")
    ax2.set_ylabel("rules in library")
    out = os.path.join(fig_dir, "cke_evolution.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return [out]


def fig_spaq_attribution(final_dir, cfg, fig_dir):
    """SPAQ 6 维属性归因：绝对误差 vs 各属性分的相关性。"""
    attr_path = os.path.join("评测数据集", "SPAQ", "spaqTest.csv")
    if not os.path.exists(attr_path):
        return []
    made = []
    for scores_path in glob.glob(os.path.join(final_dir, "*spaq*", "scores.csv")):
        run = os.path.basename(os.path.dirname(scores_path))
        with open(attr_path, encoding="utf-8-sig") as f:
            attrs = {r["image_id"]: r for r in csv.DictReader(f)}
        with open(scores_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        mos = load_mos(cfg, "spaq_test")
        dims = ["Brightness", "Colorfulness", "Contrast", "Noisiness", "Sharpness"]
        errs, dim_vals = {d: [] for d in dims}, {d: [] for d in dims}
        for r in rows:
            if r["score"] in ("", "None", None) or r["img_id"] not in mos:
                continue
            err = abs(float(r["score"]) - mos[r["img_id"]])
            for d in dims:
                errs[d].append(err)
                dim_vals[d].append(float(attrs[r["img_id"]][d]))
        corrs = [float(np.corrcoef(errs[d], dim_vals[d])[0, 1]) for d in dims]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.bar(dims, corrs)
        ax.set_ylabel("corr(attribute score, |error|)")
        ax.set_title(f"SPAQ 误差归因 — {ROUTE_LABEL.get(rows[0]['route'], run)}")
        out = os.path.join(fig_dir, f"spaq_attribution_{run}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        made.append(out)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=None, help="最终评测目录（含各路线输出与 main_table.csv）")
    args = ap.parse_args()
    cfg = get_config()
    final_dir = args.final or os.path.join(cfg.runs_dir, "final")
    fig_dir = os.path.join(final_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    made = []
    made += fig_scatter(final_dir, cfg, fig_dir)
    made += fig_ablation(final_dir, fig_dir)
    made += fig_sensitivity(cfg, fig_dir)
    made += fig_cke_evolution(cfg, fig_dir)
    made += fig_spaq_attribution(final_dir, cfg, fig_dir)
    print(f"[done] 生成 {len(made)} 张图 → {fig_dir}")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
