# -*- coding: utf-8 -*-
"""验收汇报/论文用图 v2 —— dataviz 规范：验证调色板、细标记、发丝网格、无双轴、选择性标注。
全部本地数据，0 API。输出 docs/figs/*.png。"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_mos

# ── 验证过的调色板（light, validate_palette.js 全 PASS）──
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; VIOLET = "#4a3aa7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.8, "font.size": 10,
})

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figs")
os.makedirs(OUT, exist_ok=True)
cfg = get_config()


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, labelsize=9)


def tag(ax, text):
    ax.set_title(text, fontsize=10, color=INK2, loc="left", pad=6)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(name, "ok")


# ── 图1 统一框架流程图（分支淡彩标识）──
def fig1():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.0); ax.axis("off")

    def box(x, y, w, h, text, fs=10, hue=None):
        fc = SURFACE if hue is None else hue
        alpha = 1.0 if hue is None else 0.10
        ec = BASE if hue is None else hue
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.2, alpha=alpha if hue else 1.0,
                                    mutation_aspect=1))
        if hue:  # 淡彩填充 + 彩色描边分两层
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                        fc="none", ec=hue, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color=INK2))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                    fontsize=8, color=INK2, ha="center")

    box(0.15, 2.0, 1.2, 1.0, "输入\n图像", 11)
    box(1.85, 3.55, 2.7, 1.0, "技能专家并行评分\nS-TECH / S-GLOBAL / S-CONTENT", 8.5, BLUE)
    box(1.85, 2.0, 2.7, 0.9, "像素统计特征\n（7 维, OpenCV）", 9, ORANGE)
    box(1.85, 0.5, 2.7, 0.9, "裸问释义投票\n（4 条同义问法取均值）", 9, AQUA)
    box(5.15, 2.7, 2.1, 1.25, "门控矩阵 W (3×7)\nsoftmax 动态话语权\n（BT 锦标赛离线训练）", 9, VIOLET)
    box(5.15, 0.75, 2.1, 0.8, "投票分  s_vote", 10)
    box(7.9, 1.9, 1.35, 1.0, "动态融合分\ns_fus = g·s", 10)
    box(9.75, 1.9, 1.15, 1.0, "线性混合\nα·s_fus\n+(1-α)·s_vote", 8.5)

    arrow(1.35, 2.5, 1.85, 4.0); arrow(1.35, 2.5, 1.85, 2.45); arrow(1.35, 2.5, 1.85, 0.95)
    arrow(4.55, 4.05, 5.35, 3.95, "专家分向量 s", 0.1, 0.22)
    arrow(4.55, 2.45, 5.15, 3.05, "特征 F", 0.05, -0.25)
    arrow(4.55, 0.95, 5.15, 1.15)
    arrow(7.25, 3.3, 8.25, 2.9)
    arrow(7.25, 1.15, 8.35, 1.9)
    arrow(9.25, 2.4, 9.75, 2.4)
    ax.text(10.33, 1.5, "最终质量分", fontsize=10, color=INK, ha="center")
    save(fig, "fig1_pipeline.png")


# ── 图2 门控训练曲线（小倍数图，无双轴）──
def fig2():
    steps = np.array([0, 100, 200, 300, 400, 500, 600, 700])
    loss = np.array([0.46, 0.44, 0.43, 0.43, 0.45, 0.45, 0.41, 0.40])
    wnorm = np.array([0.0, 1.3, 2.1, 2.6, 3.0, 3.3, 3.6, 3.8])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for ax, y, hue, name, ylab in [(axes[0], loss, BLUE, "(a) 训练损失  −log σ(s_A − s_B)", "损失"),
                                   (axes[1], wnorm, ORANGE, "(b) 参数范数 ‖W‖", "‖W‖")]:
        ax.plot(steps, y, color=hue, lw=2, solid_joinstyle="round")
        ax.plot(steps[-1], y[-1], "o", ms=8, mfc=hue, mec=SURFACE, mew=2)
        ax.annotate(f"{y[-1]:.2f}" if name.startswith("(a)") else f"{y[-1]:.1f}",
                    (steps[-1], y[-1]), textcoords="offset points", xytext=(-4, 10),
                    ha="right", fontsize=10, color=INK, fontweight="bold")
        tag(ax, name)
        ax.set_xlabel("训练步数"); ax.set_ylabel(ylab)
        ax.set_xlim(-20, 760)
        style(ax)
    save(fig, "fig2_loss.png")


# ── 图3 门控矩阵热力图（发散色，中性灰零点）──
def fig3():
    W = np.array([[-0.447, -0.155, 0.038, -0.429, 0, 0, 0.890],
                  [0.059, -0.010, 0.035, 0.062, 0, 0, -0.052],
                  [0.388, 0.166, -0.073, 0.367, 0, 0, -0.839]])
    rows = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
    cols = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    im = ax.imshow(W, cmap=DIV, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(7), cols, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(3), rows, fontsize=9)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(7):
            v = W[i, j]
            ax.text(j, i, f"{v:+.2f}" if v else "0", ha="center", va="center",
                    fontsize=9, color="white" if abs(v) > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("权重", color=INK2, fontsize=9)
    cb.outline.set_edgecolor(BASE); cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=3)
    save(fig, "fig3_w.png")


# ── 图4 预测分 vs 人群分（单 hue 顺序色密度）──
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, ds, eval_ds, t in [(axes[0], "koniq", "koniq_val", "(a) KonIQ-10k 验证集"),
                               (axes[1], "spaq", "spaq_test", "(b) SPAQ 测试集")]:
        pred = {}
        with open(os.path.join(cfg.runs_dir, "posthoc", f"r6_unified_{ds}", "scores.csv"),
                  encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pred[r["img_id"]] = float(r["score"])
        mos = load_mos(cfg, eval_ds)
        ids = sorted(set(pred) & set(mos))
        x = np.array([mos[i] for i in ids]); y = np.array([pred[i] for i in ids])
        ax.hexbin(x, y, gridsize=32, cmap=SEQ, mincnt=1, linewidths=0.2, edgecolors=SURFACE)
        lim = [0.8, 5.2] if ds == "koniq" else [-0.3, 10.3]
        ax.plot(lim, lim, color=MUTED, lw=1.0, solid_capstyle="round")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("人群主观分 MOS"); ax.set_ylabel("框架预测分")
        srcc = float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(x, y).statistic)
        ax.annotate(f"SRCC = {srcc:.3f}", (0.04, 0.93), xycoords="axes fraction",
                    fontsize=10, color=INK, fontweight="bold")
        tag(ax, t)
        style(ax, ygrid=False)
    save(fig, "fig4_scatter.png")


# ── 图5 提示词约束消融（分类色分组柱，选择性标注）──
def fig5():
    arms = ["R1-bare\n(纯裸问)", "R1-anchor v2\n(轻人设+锚点)", "R1-anchor v3\n(+维度/清单/程序)", "R1-rich\n(完整专家)"]
    k_srcc = [0.660, 0.612, 0.625, 0.633]; s_srcc = [0.884, 0.889, 0.865, 0.867]
    k_mae = [0.454, 0.507, 0.645, 0.688];   s_mae = [0.819, 1.034, 1.062, 1.042]
    x = np.arange(4); w = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, kv, sv, name, ylab, ylim, labels in [
            (axes[0], k_srcc, s_srcc, "(a) 排序一致性 SRCC ↑", "SRCC", (0, 1.0), False),
            (axes[1], k_mae, s_mae, "(b) 绝对误差 MAE ↓", "MAE", (0, 1.25), True)]:
        b1 = ax.bar(x - (w + 0.03) / 2, kv, w, color=BLUE, label="KonIQ",
                    edgecolor=SURFACE, linewidth=1.5)
        b2 = ax.bar(x + (w + 0.03) / 2, sv, w, color=ORANGE, label="SPAQ",
                    edgecolor=SURFACE, linewidth=1.5)
        if labels:  # 只标故事两端：纯裸问与完整专家
            for rect in [b1[0], b1[-1], b2[0], b2[-1]]:
                ax.annotate(f"{rect.get_height():.3f}",
                            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(x, arms, fontsize=8.5)
        ax.set_ylim(*ylim); ax.set_ylabel(ylab)
        tag(ax, name)
        ax.legend(frameon=False, fontsize=9, loc="upper left" if not labels else "upper left")
        style(ax)
    save(fig, "fig5_ablation.png")


# ── 图6 逐臂趋势（R1→R6 演进）──
def fig6():
    arms = ["R1-bare\n(锚点基准)", "R2", "R2.5", "R3", "R6\n统一框架"]
    k_srcc = [0.6246, 0.6061, 0.6193, 0.6207, 0.7335]
    s_srcc = [0.8652, 0.8591, 0.8602, 0.8607, 0.8910]
    k_mae = [0.6446, 0.6773, 0.5333, 0.4994, 0.4910]
    s_mae = [1.0618, 0.9714, 1.0138, 1.0080, 0.7963]
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, kv, sv, name, ylab in [(axes[0], k_srcc, s_srcc, "(a) SRCC ↑", "SRCC"),
                                   (axes[1], k_mae, s_mae, "(b) MAE ↓", "MAE")]:
        for vals, hue, lab in [(kv, BLUE, "KonIQ"), (sv, ORANGE, "SPAQ")]:
            ax.plot(x, vals, color=hue, lw=2, label=lab, zorder=2,
                    marker="o", ms=7, mfc=hue, mec=SURFACE, mew=2)
            ax.annotate(f"{vals[-1]:.3f}", (x[-1], vals[-1]), xytext=(6, -3),
                        textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xticks(x, arms, fontsize=9)
        ax.set_ylabel(ylab)
        ax.margins(x=0.08)
        tag(ax, name)
        ax.legend(frameon=False, fontsize=9, loc="best")
        style(ax)
    save(fig, "fig6_progression.png")


# ── 图7 门控跨模型迁移（棒棒糖 + 参考线）──
def fig7():
    labels = ["8B 纯裸问", "8B + 等权门控", "8B + 迁移 32B 门控"]
    vals = [0.6789, 0.7587, 0.7764]
    ref = 0.7335
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.vlines(x, 0.60, vals, color=BLUE, lw=2)
    ax.plot(x, vals, "o", ms=11, mfc=BLUE, mec=SURFACE, mew=2, zorder=3)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.axhline(ref, color=INK2, lw=1.0)
    ax.annotate(f"32B 统一框架 = {ref:.3f}", (2.35, ref), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_ylim(0.60, 0.82)
    ax.set_ylabel("KonIQ SRCC ↑")
    ax.set_xlim(-0.4, 2.6)
    tag(ax, "门控矩阵跨模型迁移（0 新增调用）")
    style(ax)
    save(fig, "fig7_transfer.png")


# ── 图8 分布框架效应（MOS vs 纯裸问 vs 完整专家）──
def fig8():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    conf = [(axes[0], "koniq", "koniq_val", np.arange(1, 5.001, 0.25), "(a) KonIQ-10k 验证集", "分数（1–5）"),
            (axes[1], "spaq", "spaq_test", np.arange(0, 10.001, 0.5), "(b) SPAQ 测试集", "分数（0–10）")]
    for ax, ds, eval_ds, bins, t, xl in conf:
        mos = load_mos(cfg, eval_ds)
        series = [("人群分 MOS", MUTED, np.array(list(mos.values())))]
        for route, lab, hue in [("r1b", "R1-bare（纯裸问）", BLUE), ("r1r", "R1-rich（完整专家）", ORANGE)]:
            pred = []
            with open(os.path.join(cfg.runs_dir, "final", f"{route}_{ds}", "scores.csv"),
                      encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r.get("score") not in ("", None):
                        pred.append(float(r["score"]))
            series.append((lab, hue, np.array(pred)))
        for i, (lab, hue, arr) in enumerate(series):
            dens, edges = np.histogram(arr, bins=bins, density=True)
            ax.stairs(dens, edges, color=hue, lw=2, label=lab)
            if i == 0:
                ax.stairs(dens, edges, fill=True, color=hue, alpha=0.15)
        ax.set_xlabel(xl); ax.set_ylabel("密度")
        tag(ax, t)
        ax.margins(y=0.22)
        ax.legend(frameon=False, fontsize=9,
                  loc="upper left" if ds == "koniq" else "upper center")
        style(ax)
    save(fig, "fig8_dist.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("all ->", OUT)
