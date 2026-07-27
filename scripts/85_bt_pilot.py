# -*- coding: utf-8 -*-
"""二轮 S0：BT 试赛（300 图统一锦标赛 + 门控）。

合规（ADR-0003）：仅 KonIQ Train 像素，零 MOS；文件名不进 prompt。

设计（docs/二轮重设计计划-通俗版.md §3.2/§10.2/§11.2）：
- 节点：300 张试赛图（CKE 工作集中按旧分十分位分层抽样，旧分=一轮缓存回放，零新打分）
       + 48 个源各自 2 张程序失真后代（jpeg q10 / down ×4，共 96 张，本地造，排序已知）。
- 边（防"旧分回声"循环，§10.1）：旧分相邻对×双序 + 随机对 + 专家分歧对 + OpenCV 难分对
       + 后代↔自然邻对 + 已知真理探针对；梯子已知边（源>后代）权重 2。
- 门控（不过则弃 BT，Router 只练梯子）：
  G1 探针：已知真理对 VLM 两两准确率 ≥0.90（pairwise 信号本身的体检）；
  G2 主门控：留出对决预测（5 折），BT 隐分预测胜者准确率 > 旧分 +0.02；
  G3 稳定：两半边分别拟合，SRCC ≥0.75；另报 BT-旧分 SRCC（回声检查）。
  G2b 诊断：源级 CV 的梯子已知边排序（只报告，不作门控）。

用法：
  python scripts/85_bt_pilot.py --dry-run   # 缓存回放旧分（免费）+ 本地步骤，不碰新 API
  python scripts/85_bt_pilot.py --sim       # 合成数据验证 BT/门控数学（不碰 API）
  python scripts/85_bt_pilot.py --limit 40  # 冒烟：只跑少量对（~¥0.1）
  python scripts/85_bt_pilot.py             # 正式试赛（~3k 次调用，≈¥5）
"""
import argparse
import asyncio
import io
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.cke import _parse_winner
from iqa_agent.data import load_images
from iqa_agent.pipeline import run_r2
from iqa_agent.prompts.pairwise import build_pairwise_prompt

DESC_FAMILIES = ["jpeg", "down"]          # 难族（F-001：jpeg 是盲区）；down=缩放×4
N_PILOT, N_DESC_SRC = 300, 48
KNOWN_W = 2.0                              # 已知真理边权重
# 边密度（sim 实测：稀疏链图上 BT 恢复力崩坏；随机长程边 ~2000 时 BT 才超过旧分 +0.05）
N_RANDOM, N_SPREAD, N_OCV = 2000, 150, 100


# ---------- 本地：后代生成 ----------

def gen_descendants(pilot, img_dir):
    """48 源 × 2 族 = 96 张后代。返回 [{node, src_pos, family, path}]。"""
    os.makedirs(img_dir, exist_ok=True)
    desc = []
    src_positions = list(range(0, len(pilot), 6))[:N_DESC_SRC]
    for k, pos in enumerate(src_positions):
        img = Image.open(pilot[pos]["path"]).convert("RGB")
        for f, fam in enumerate(DESC_FAMILIES):
            if fam == "jpeg":
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=10)
                buf.seek(0)
                out = Image.open(buf).convert("RGB")
            else:
                w, h = img.size
                out = img.resize((w // 4, h // 4), Image.BICUBIC).resize((w, h), Image.BICUBIC)
            node = 300 + 2 * k + f
            path = os.path.join(img_dir, f"{node:03d}_{fam}.jpg")
            out.save(path, "JPEG", quality=95)
            desc.append({"node": node, "src_pos": pos, "family": fam, "path": path})
    return desc


# ---------- 本地：OpenCV 手工特征（numpy，零 API，外援信号 C） ----------

def opencv_features(path: str) -> dict:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
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
    return {"lap_var": lap_var, "noise": noise, "colorful": colorful, "bright": float(gray.mean())}


# ---------- 配对选择 ----------

def build_pairs(pilot, desc, feats, seed):
    rng = random.Random(seed)
    n = len(pilot)
    order = sorted(range(n), key=lambda i: pilot[i]["score"])
    pairs = []

    def add(i, j, kind, orders=1):
        pairs.append({"i": i, "j": j, "kind": kind, "orders": orders})

    used = set()
    for r in range(n - 1):  # 1) 旧分相邻对 ×双序
        a, b = order[r], order[r + 1]
        used.add((min(a, b), max(a, b)))
        add(a, b, "adjacent", orders=2)
    got = 0  # 2) 随机对（长程骨架，密度经 sim 标定）
    while got < N_RANDOM:
        a, b = rng.sample(range(n), 2)
        key = (min(a, b), max(a, b))
        if key not in used:
            used.add(key)
            add(a, b, "random")
            got += 1
    iqr = []  # 3) 专家分歧对
    for p in pilot:
        ss = [v for v in p["skill_scores"].values() if v is not None]
        q = np.percentile(ss, [25, 75]) if len(ss) >= 2 else [0, 0]
        iqr.append(float(q[1] - q[0]))
    cand = [(i, j) for i in range(n) for j in range(i + 1, n)
            if abs(pilot[i]["score"] - pilot[j]["score"]) <= 0.15 and (i, j) not in used]
    cand.sort(key=lambda t: -(iqr[t[0]] + iqr[t[1]]))
    for a, b in cand[:N_SPREAD]:
        used.add((a, b))
        add(a, b, "spread")
    laps = np.array([feats[str(i)]["lap_var"] for i in range(n)])  # 4) OpenCV 难分对
    m, s = laps.mean(), laps.std() + 1e-9
    cand2 = [(i, j) for i in range(n) for j in range(i + 1, n)
             if abs(pilot[i]["score"] - pilot[j]["score"]) <= 0.15 and (i, j) not in used]
    cand2.sort(key=lambda t: -abs((laps[t[0]] - m) / s - (laps[t[1]] - m) / s))
    for a, b in cand2[:N_OCV]:
        used.add((a, b))
        add(a, b, "ocv_hard")
    rank_of = {idx: r for r, idx in enumerate(order)}  # 5) 后代↔自然邻对 ×3（保连通，G2b 用）
    for d in desc:
        r = rank_of[d["src_pos"]]
        for k, dr in enumerate((1, -1, 2)):
            rr = min(max(r + dr, 0), n - 1)
            if rr != r:
                add(d["node"], order[rr], "desc_nat")
    for d in desc:  # 6) 已知真理探针对（前 24 源）×双序
        if d["family"] == "jpeg" and d["src_pos"] // 6 < 24:
            add(d["src_pos"], d["node"], "probe", orders=2)
    return pairs


# ---------- BT ----------

def fit_bt_vec(wins: np.ndarray, iters: int = 300) -> np.ndarray:
    """Hunter MM（与 cke.fit_bt 同算法，向量化）。注意：在链式稀疏图上收敛极慢，
    只作对照保留；正式拟合用 fit_bt_nll。"""
    games = wins + wins.T
    wsum = wins.sum(axis=1)
    w = np.ones(wins.shape[0])
    for _ in range(iters):
        for i in range(wins.shape[0]):
            denom = float(np.sum(games[i] / (w[i] + w)))
            if denom > 0:
                w[i] = wsum[i] / denom if wsum[i] > 0 else 1e-12
        w /= np.exp(np.mean(np.log(w + 1e-12)))
    return np.log(w + 1e-12)


def fit_bt_nll(wins: np.ndarray) -> np.ndarray:
    """L-BFGS 直接优化 BT 对数强度的凸负对数似然（链式稀疏图也能快速收敛）。
    NLL(θ) = -Σ_ij wins[i,j]·[θ_i − log(exp θ_i + exp θ_j)] + 1e-6·Σθ²（微弱先验破尺度未定）。
    """
    from scipy.optimize import minimize
    n = wins.shape[0]
    idx_i, idx_j = np.nonzero(wins)
    w_ij = wins[idx_i, idx_j]
    g_ij = w_ij + wins[idx_j, idx_i]  # 该对总场数

    def nll_grad(theta):
        # logaddexp 数值稳定形式：log(ei+ej) = logaddexp(θ_i, θ_j)，p = exp(θ_i - logaddexp)
        la = np.logaddexp(theta[idx_i], theta[idx_j])
        p = np.exp(theta[idx_i] - la)
        ll = -float(np.sum(w_ij * theta[idx_i])) + float(np.sum(g_ij * la)) \
            + 1e-6 * float(np.sum(theta ** 2))
        grad = np.zeros(n)
        np.add.at(grad, idx_i, -w_ij + g_ij * p)
        np.add.at(grad, idx_j, g_ij * (1 - p))
        grad += 2e-6 * theta
        return ll, grad

    res = minimize(nll_grad, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-12})
    return res.x


def fit_bt(wins: np.ndarray) -> np.ndarray:
    """正式入口：NLL 拟合（返回对数强度，越大越好）。"""
    return fit_bt_nll(wins)


def build_wins(duels, desc, n_nodes, holdout_srcs=None):
    wins = np.zeros((n_nodes, n_nodes))
    for d in duels:
        if d.get("winner_node") is not None:
            wins[d["winner_node"], d["loser_node"]] += 1
    for dd in desc:
        if holdout_srcs and dd["src_pos"] in holdout_srcs:
            continue
        wins[dd["src_pos"], dd["node"]] += KNOWN_W  # 已知真理：源 > 后代
    return wins


# ---------- 门控计算（真数据与 sim 共用） ----------

def compute_gates(duels, desc, pilot_scores, desc_scores, seed):
    """pilot_scores: 300 个旧分；desc_scores: {str(node): 旧分}。返回 report dict。

    G1 探针：已知真理对（源 vs 自家 jpeg 后代）VLM 两两准确率 ≥0.90；
    G2 主门控：留出对决预测——5 折留出自然-自然对决，BT 隐分预测胜者 vs 旧分预测胜者，
       BT 准确率须 > 旧分 +0.02（直接回答"BT 是否比旧分包含更多成对质量信息"）；
    G2b 诊断：源级 5 折 CV，BT 在留出已知边（源>后代）上的排序准确率 vs 旧分（只做报告）；
    G3 稳定：两半边拟合 SRCC ≥0.75；另报 BT-旧分 SRCC（回声检查）。
    """
    n_nodes = 300 + len(desc)
    known_pairs = [(d["src_pos"], d["node"], d["family"]) for d in desc]

    # ---- G1 探针 ----
    probe = [d for d in duels if d["kind"] == "probe" and d["winner_node"] is not None]
    g1 = sum(1 for d in probe if d["winner_node"] == d["i"]) / len(probe) if probe else 0.0

    # ---- G2 留出对决预测（自然-自然对决 5 折）----
    nn = [d for d in duels if d["winner_node"] is not None
          and d["i"] < 300 and d["j"] < 300 and d["kind"] != "probe"]
    random.Random(seed + 1).shuffle(nn)
    folds = [nn[k::5] for k in range(5)]
    bt_hits = old_hits = tot = 0
    for k in range(5):
        holdout = folds[k]
        hold_keys = {(d["i"], d["j"], d["winner_node"]) for d in holdout}
        train = [d for d in duels if (d["i"], d["j"], d["winner_node"]) not in hold_keys]
        lat = fit_bt(build_wins(train, desc, n_nodes))
        for d in holdout:
            pred_bt = d["i"] if lat[d["i"]] > lat[d["j"]] else d["j"]
            pred_old = d["i"] if pilot_scores[d["i"]] > pilot_scores[d["j"]] else d["j"]
            bt_hits += pred_bt == d["winner_node"]
            old_hits += pred_old == d["winner_node"]
            tot += 1
    g2_bt = bt_hits / tot if tot else 0.0
    g2_old = old_hits / tot if tot else 0.0

    # ---- G2b 源级 CV（诊断，不作门控）----
    srcs = sorted({d["src_pos"] for d in desc})
    cv_hits = cv_old = cv_tot = 0
    fam_stat = {}
    for k in range(5):
        fold = set(srcs[k::5])
        lat = fit_bt(build_wins(duels, desc, n_nodes, holdout_srcs=fold))
        for u, v, fam in known_pairs:
            if u not in fold:
                continue
            ds = desc_scores.get(str(v))
            if ds is None:
                continue
            cv_hits += lat[u] > lat[v]
            cv_old += pilot_scores[u] > ds
            cv_tot += 1
            fam_stat.setdefault(fam, [0, 0, 0])
            fam_stat[fam][0] += lat[u] > lat[v]
            fam_stat[fam][1] += pilot_scores[u] > ds
            fam_stat[fam][2] += 1
    g2b_bt = cv_hits / cv_tot if cv_tot else 0.0
    g2b_old = cv_old / cv_tot if cv_tot else 0.0

    # ---- G3 两半稳定性 + 回声 ----
    judged = [d for d in duels if d["winner_node"] is not None]
    random.Random(seed).shuffle(judged)
    lat_a = fit_bt(build_wins(judged[: len(judged) // 2], desc, n_nodes))
    lat_b = fit_bt(build_wins(judged[len(judged) // 2:], desc, n_nodes))
    g3 = float(spearmanr(lat_a[:300], lat_b[:300]).statistic)
    full_lat = fit_bt(build_wins(duels, desc, n_nodes))
    echo = float(spearmanr(full_lat[:300], pilot_scores).statistic)

    report = {
        "G1_probe_acc": round(float(g1), 4), "G1_pass": bool(g1 >= 0.90),
        "G2_bt_holdout_acc": round(float(g2_bt), 4), "G2_old_holdout_acc": round(float(g2_old), 4),
        "G2_pass": bool(g2_bt > g2_old + 0.02),
        "G2b_src_cv_bt": round(float(g2b_bt), 4), "G2b_src_cv_old": round(float(g2b_old), 4),
        "G2b_by_family": {f: {"bt": round(a / t, 3), "old": round(o / t, 3), "n": t}
                          for f, (a, o, t) in fam_stat.items()},
        "G3_split_half_srcc": round(float(g3), 4), "G3_pass": bool(g3 >= 0.75),
        "echo_srcc_bt_vs_old": round(float(echo), 4),
        "n_duels": len(duels), "n_invalid": sum(1 for d in duels if d["winner_node"] is None),
    }
    verdict = report["G1_pass"] and report["G2_pass"] and report["G3_pass"]
    report["verdict"] = "GO（可上全量锦标赛）" if verdict else "NO-GO（Router 只练梯子信号）"
    return report


# ---------- sim：合成数据验证数学（不碰 API、不碰图像） ----------

def run_sim(cfg):
    """两种噪声体制验证：温和（判噪 0.03，门控应全过）+ 严苛（判噪 0.10，G2 应拒绝）。"""
    for label, jnoise in [("温和", 0.03), ("严苛", 0.10)]:
        rng = np.random.default_rng(7)
        n = 300
        true = rng.uniform(0, 1, n)                       # 真实质量（合成）
        old = true + rng.normal(0, 0.15, n)               # 旧分 = 含噪真理
        src_pos = list(range(0, n, 6))[:N_DESC_SRC]
        desc = [{"node": 300 + 2 * k + f, "src_pos": p, "family": DESC_FAMILIES[f]}
                for k, p in enumerate(src_pos) for f in range(2)]
        true_full = np.concatenate([true, [true[d["src_pos"]] - 0.5 for d in desc]])

        order = np.argsort(old)
        duels = []

        def judge(a, b, kind):
            wa, wb = true_full[a] + rng.normal(0, jnoise), true_full[b] + rng.normal(0, jnoise)
            w = a if wa > wb else b
            duels.append({"i": a, "j": b, "kind": kind,
                          "winner_node": w, "loser_node": b if w == a else a})

        for r in range(n - 1):                            # 相邻对 ×双序
            judge(int(order[r]), int(order[r + 1]), "adjacent")
            judge(int(order[r + 1]), int(order[r]), "adjacent")
        for _ in range(N_RANDOM):                         # 随机对（长程骨架）
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a != b:
                judge(a, b, "random")
        for d in desc:                                    # 后代邻对 ×3
            r = int(np.where(order == d["src_pos"])[0][0])
            for dr in (1, -1, 2):
                judge(d["node"], int(order[min(max(r + dr, 0), n - 1)]), "desc_nat")
        for d in desc[:24]:                               # 探针 ×双序
            judge(d["src_pos"], d["node"], "probe")
            judge(d["src_pos"], d["node"], "probe")

        pilot_scores = old.tolist()
        desc_scores = {str(d["node"]): float(true_full[d["node"]] + rng.normal(0, 0.15)) for d in desc}
        report = compute_gates(duels, desc, pilot_scores, desc_scores, cfg.seed)
        bt_lat = fit_bt(build_wins(duels, desc, 300 + len(desc)))
        report["sim_recovery_srcc_bt_vs_true"] = round(float(spearmanr(bt_lat[:300], true).statistic), 4)
        report["sim_recovery_srcc_old_vs_true"] = round(float(spearmanr(old, true).statistic), 4)
        print(f"===== sim[{label}] 判噪={jnoise} =====")
        print(json.dumps(report, ensure_ascii=False, indent=1))
    print("[sim] 验证标准：温和体制门控应全过（否则代码有 bug）；严苛体制 G2 应拒绝（说明门控能识别低质量信号）")


# ---------- 主流程 ----------

async def main_async(args):
    cfg = get_config()
    if args.sim:
        run_sim(cfg)
        return
    wd = os.path.join(cfg.runs_dir, "bt_pilot")
    img_dir = os.path.join(wd, "images")
    os.makedirs(wd, exist_ok=True)
    client = VLMClient(cfg, cfg.model_main)

    # ---- Stage 0: 工作集旧分（一轮缓存回放，零新打分；未命中则报警中止）----
    with open(os.path.join(cfg.ladder_dir, "source_ids.json")) as f:
        ladder_src = set(json.load(f))
    ws_path = os.path.join(wd, "workset_scores.json")
    rows = None
    if os.path.exists(ws_path) and not args.refresh:
        rows = json.load(open(ws_path, encoding="utf-8"))
        print(f"[S0] 工作集旧分已落盘（{len(rows)} 行），跳过回放")
    if rows is None:
        workset = load_images(cfg, "koniq_train", limit=cfg.workset_size,
                              seed=cfg.workset_seed, exclude=ladder_src)
        rows = await run_r2(client, workset, "koniq", cfg.scales["koniq"],
                            dynamic=False, rules_by_skill=None)
        if client.calls > 100:
            print(f"⛔ [S0] 缓存未命中（新调用 {client.calls} 次）！检查 prompt/路径是否与一轮 CKE 一致。中止防烧钱。")
            return
        rows = [dict(r, path=img.path) for r, img in zip(rows, workset)]
        json.dump(rows, open(ws_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[S0] 工作集旧分回放完成（api_calls={client.calls}，应为 0）")

    # ---- Stage 1: 试赛 300 张（旧分十分位分层）----
    valid = [r for r in rows if r.get("score") is not None]
    valid.sort(key=lambda r: r["score"])
    rng = random.Random(cfg.seed)
    pilot = []
    for b in range(10):
        bin_rows = valid[int(len(valid) * b / 10): int(len(valid) * (b + 1) / 10)]
        pilot.extend(rng.sample(bin_rows, min(N_PILOT // 10, len(bin_rows))))
    for p in pilot:
        if isinstance(p["skill_scores"], str):
            p["skill_scores"] = json.loads(p["skill_scores"])
    json.dump([p["img_id"] for p in pilot], open(os.path.join(wd, "pilot_ids.json"), "w"))
    print(f"[S1] 试赛 {len(pilot)} 张（分数范围 {min(p['score'] for p in pilot):.2f}~{max(p['score'] for p in pilot):.2f}）")

    # ---- Stage 2-3: 后代 + OpenCV 特征（本地，零 API）----
    desc = gen_descendants(pilot, img_dir)
    fpath = os.path.join(wd, "features.json")
    if os.path.exists(fpath) and not args.refresh:
        feats = json.load(open(fpath))
    else:
        feats = {}
        for i, p in enumerate(pilot):
            feats[str(i)] = opencv_features(p["path"])
        for d in desc:
            feats[str(d["node"])] = opencv_features(d["path"])
        json.dump(feats, open(fpath, "w"))
    print(f"[S2-3] 后代 {len(desc)} 张，特征 {len(feats)} 份")

    # ---- Stage 4: 配对清单 ----
    pairs = build_pairs(pilot, desc, feats, cfg.seed)
    n_calls = sum(p["orders"] for p in pairs)
    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + p["orders"]
    print(f"[S4] 配对 {len(pairs)} 组 → {n_calls} 次调用 {kinds}")
    json.dump(pairs, open(os.path.join(wd, "pairs.json"), "w"))
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"  [--limit] 只跑前 {len(pairs)} 组（冒烟模式，门控结果无意义）")
    if args.dry_run:
        print("[dry-run] 到此为止，未碰新 API。")
        return

    # ---- Stage 5: 两两对比 ----
    desc_by_node = {d["node"]: d for d in desc}

    def node_path(node):
        return pilot[node]["path"] if node < 300 else desc_by_node[node]["path"]

    prompt = build_pairwise_prompt()
    duel_jobs = []
    for p in pairs:
        for o in range(p["orders"]):
            first, second = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
            duel_jobs.append({"i": p["i"], "j": p["j"], "first": first, "second": second, "kind": p["kind"]})

    async def duel(job):
        text, _ = await client.compare_images(node_path(job["first"]), node_path(job["second"]), prompt)
        w = _parse_winner(text)
        if w == "A":
            job["winner_node"], job["loser_node"] = job["first"], job["second"]
        elif w == "B":
            job["winner_node"], job["loser_node"] = job["second"], job["first"]
        else:
            job["winner_node"] = job["loser_node"] = None
        return job

    dpath = os.path.join(wd, "duels.json")
    duels = json.load(open(dpath, encoding="utf-8")) if os.path.exists(dpath) and not args.refresh else []
    done_keys = {(d["i"], d["j"], d["first"]) for d in duels}
    todo = [j for j in duel_jobs if (j["i"], j["j"], j["first"]) not in done_keys]
    print(f"[S5] 两两对比 {len(todo)} 场（已完成 {len(done_keys)}）")
    results = await gather_with_progress([duel(j) for j in todo], every=200, label="bt-duels")
    n_bad = sum(1 for r in results if isinstance(r, Exception) or r["winner_node"] is None)
    duels.extend([r for r in results if not isinstance(r, Exception)])
    json.dump(duels, open(dpath, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[S5] 完成，无效 {n_bad} 场；账本 {client.ledger()}")

    # ---- Stage 6: 后代旧分（5 专家融合，供门控对照）----
    ds_path = os.path.join(wd, "desc_scores.json")
    desc_scores = json.load(open(ds_path)) if os.path.exists(ds_path) else {}
    todo_desc = [d for d in desc if str(d["node"]) not in desc_scores]
    if todo_desc:
        refs = [SimpleNamespace(img_id=str(d["node"]), path=d["path"], dataset="koniq") for d in todo_desc]
        rrows = await run_r2(client, refs, "koniq", cfg.scales["koniq"], dynamic=False, rules_by_skill=None)
        for d, r in zip(todo_desc, rrows):
            desc_scores[str(d["node"])] = r["score"]
        json.dump(desc_scores, open(ds_path, "w"))
        print(f"[S6] 后代旧分 {len(todo_desc)} 张；账本 {client.ledger()}")

    # ---- Stage 7: 门控 ----
    report = compute_gates(duels, desc, [p["score"] for p in pilot], desc_scores, cfg.seed)
    report["ledger"] = client.ledger()
    json.dump(report, open(os.path.join(wd, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="缓存回放（免费）+ 本地步骤，不碰新 API")
    ap.add_argument("--sim", action="store_true", help="合成数据验证 BT/门控数学")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 组配对（冒烟）")
    ap.add_argument("--refresh", action="store_true", help="忽略中间产物重来（缓存调用仍免费）")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
