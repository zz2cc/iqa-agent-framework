# -*- coding: utf-8 -*-
"""二轮 S3：全量统一锦标赛（KonIQ 1,000 节点 / SPAQ 400 节点）。

合规（ADR-0003）：仅 Train 像素，零 MOS；SPAQ 训练信号 1568px 降采样协议（已声明）。

设计（S0 试赛 GO 后的放量，runs/bt_pilot/report.json）：
- KonIQ：CKE 工作集 1,000 张（旧分=一轮 5 专家融合，缓存免费）+ 160 源 × 2 族后代
  （blur σ4 / noise σ50，KonIQ 梯子体检端点 ~1.0 的最可靠族，F-015）；已知边权重 2。
- SPAQ：400 张（150 个梯子源【S-TECH+S-GLOBAL 旧分已有】+ 250 新图【补评 2 技能】）；
  后代复用梯子 2.0 的 blur_L2 / jpeg_L2（F-016 探针验证 0.98/0.88；down 0.72 弃用）。
- 边密度按 F-013 标定：随机长程边 ~6-8 条/节点为骨架，相邻对局部微调，
  分歧/OpenCV 对破回声，后代边接已知真理，探针测两两可靠性。
- 门控同试赛：G1 ≥0.90、G2 BT 留出对决 > 旧分 +0.02、G3 ≥0.75。

用法：
  python scripts/86_full_tournament.py --domain koniq --dry-run
  python scripts/86_full_tournament.py --domain koniq
  python scripts/86_full_tournament.py --domain spaq
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
from PIL import Image, ImageFilter
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.client import VLMClient, gather_with_progress
from iqa_agent.cke import _parse_winner
from iqa_agent.pipeline import run_r2
from iqa_agent.prompts.pairwise import build_pairwise_prompt

KNOWN_W = 2.0
WORKDIR = os.path.join("runs", "full_tournament")


# ---------- 图像与特征 ----------

def resize_to(src, dst, max_side):
    img = Image.open(src).convert("RGB")
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    img.save(dst, "JPEG", quality=95)
    return dst


def opencv_features(path: str) -> dict:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    c = gray[1:-1, 1:-1]
    lap = -4 * c + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return {"lap_var": float(np.var(lap))}


def gen_desc(path, fam, out_path):
    img = Image.open(path).convert("RGB")
    if fam == "blur":
        out = img.filter(ImageFilter.GaussianBlur(radius=4))
    elif fam == "noise":
        arr = np.asarray(img, dtype=np.float32)
        rng = np.random.default_rng(42)
        out = Image.fromarray(np.clip(arr + rng.normal(0, 50, arr.shape), 0, 255).astype(np.uint8))
    else:
        raise ValueError(fam)
    out.save(out_path, "JPEG", quality=95)


# ---------- BT ----------

def fit_bt(wins: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize
    n = wins.shape[0]
    idx_i, idx_j = np.nonzero(wins)
    w_ij = wins[idx_i, idx_j]
    g_ij = w_ij + wins[idx_j, idx_i]

    def nll_grad(theta):
        la = np.logaddexp(theta[idx_i], theta[idx_j])
        p = np.exp(theta[idx_i] - la)
        ll = -float(np.sum(w_ij * theta[idx_i])) + float(np.sum(g_ij * la)) + 1e-6 * float(np.sum(theta ** 2))
        grad = np.zeros(n)
        np.add.at(grad, idx_i, -w_ij + g_ij * p)
        np.add.at(grad, idx_j, g_ij * (1 - p))
        grad += 2e-6 * theta
        return ll, grad

    res = minimize(nll_grad, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": 5000})
    return res.x


def build_wins(duels, desc, n_nodes, holdout_srcs=None):
    wins = np.zeros((n_nodes, n_nodes))
    for d in duels:
        if d.get("winner_node") is not None:
            wins[d["winner_node"], d["loser_node"]] += 1
    for dd in desc:
        if holdout_srcs and dd["src_node"] in holdout_srcs:
            continue
        wins[dd["src_node"], dd["node"]] += KNOWN_W
    return wins


# ---------- 节点装配 ----------

def load_koniq_nodes(cfg, wd):
    rows = json.load(open(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"), encoding="utf-8"))
    nodes = [{"node": i, "path": r["path"], "old": r["score"],
              "skill_scores": json.loads(r["skill_scores"]) if isinstance(r["skill_scores"], str) else r["skill_scores"]}
             for i, r in enumerate(rows) if r.get("score") is not None]
    print(f"[koniq] 节点 {len(nodes)} 张（旧分 5 专家融合，免费）")
    return nodes


def load_spaq_nodes(cfg, wd, client, dry_run):
    d2 = os.path.join(cfg.runs_dir, "ladder2")
    scores = json.load(open(os.path.join(d2, "scores.json")))
    # 复现梯子 SPAQ 源挑选（同 seed）：src_idx 300+k ↔ sorted(pick)[k]
    spaq_dir = os.path.join(cfg.runs_dir, "spaq_train_images")
    spaq_files = sorted(f for f in os.listdir(spaq_dir) if f.lower().endswith(".jpg"))
    rng_p = np.random.default_rng(cfg.seed + 2000)
    pick150 = sorted(rng_p.choice(spaq_files, size=150, replace=False))
    l2_manifest = json.load(open(os.path.join(d2, "manifest.json")))
    orig_by_src = {m["src_idx"]: m["file"] for m in l2_manifest if m["domain"] == "spaq" and m["family"] == "orig"}

    stage = os.path.join(wd, "spaq_imgs")
    os.makedirs(stage, exist_ok=True)
    nodes = []
    for k, fn in enumerate(pick150):  # 前 150：梯子源，旧分 = 2 技能均值（已有）
        src_idx = 300 + k
        f2 = orig_by_src[src_idx]
        sc = scores.get(f2, {})
        old = float(np.mean([sc["S-TECH"], sc["S-GLOBAL"]])) if "S-TECH" in sc and "S-GLOBAL" in sc else None
        nodes.append({"node": len(nodes), "path": os.path.join(d2, "images", f2),
                      "old": old, "skill_scores": sc, "ladder_src": src_idx, "spaq_fn": fn})
    rest = [f for f in spaq_files if f not in set(pick150)]
    rng_n = np.random.default_rng(cfg.seed + 3000)
    pick250 = sorted(rng_n.choice(rest, size=250, replace=False))
    todo = []
    for fn in pick250:  # 后 250：新图，先降采样暂存；旧分由调用方补评
        dst = os.path.join(stage, fn)
        if not os.path.exists(dst):
            resize_to(os.path.join(spaq_dir, fn), dst, 1568)
        nodes.append({"node": len(nodes), "path": dst, "old": None, "skill_scores": {}, "spaq_fn": fn})
        todo.append(nodes[-1])
    return nodes, todo


# ---------- 主流程 ----------

async def main_async(args):
    cfg = get_config()
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    os.makedirs(wd, exist_ok=True)
    dom = args.domain
    client = VLMClient(cfg, cfg.model_main)

    # ---- 节点 ----
    if dom == "koniq":
        nodes = load_koniq_nodes(cfg, wd)
        N = len(nodes)
        n_random, n_spread, n_ocv, adj_orders = 8000, 400, 300, 2  # G3 两半密度 ≥6/节点（F-013）
        desc_fams = ["blur", "noise"]
        n_desc_src = 160
    else:
        nodes, todo_new = load_spaq_nodes(cfg, wd, client, args.dry_run)
        N = len(nodes)
        n_random, n_spread, n_ocv, adj_orders = 3000, 0, 0, 1  # G3 密度补强（+1,800 随机边）
        desc_fams = ["blur", "jpeg"]
        n_desc_src = 100
        if todo_new and not args.dry_run:
            from iqa_agent.prompts.skills import build_skill_prompt
            from iqa_agent.scoring import parse_score

            async def score_one(n_, sk):
                text, _ = await client.score_image(n_["path"], build_skill_prompt(sk, "spaq", rules=None), temperature=0.0)
                p = parse_score(text, cfg.scales["spaq"])
                return n_["node"], sk, (p["score"] if p else None)

            jobs = [score_one(n_, sk) for n_ in todo_new for sk in ("S-TECH", "S-GLOBAL")]
            rows = await gather_with_progress(jobs, every=200, label="spaq-old-score")
            for r in rows:
                if not isinstance(r, Exception) and r[2] is not None:
                    nodes[r[0]]["skill_scores"][r[1]] = r[2]
            for n_ in nodes:
                sc = n_["skill_scores"]
                if n_["old"] is None and "S-TECH" in sc and "S-GLOBAL" in sc:
                    n_["old"] = float(np.mean([sc["S-TECH"], sc["S-GLOBAL"]]))
            print(f"[spaq] 新图旧分补齐 {len(todo_new)} 张；账本 {client.ledger()}")
    nodes = [n_ for n_ in nodes if n_.get("old") is not None]
    for i, n_ in enumerate(nodes):
        n_["node"] = i  # 重排为连续节点号
    N = len(nodes)
    json.dump([{k: v for k, v in n_.items() if k != "skill_scores"} for n_ in nodes],
              open(os.path.join(wd, f"nodes_{dom}.json"), "w"), ensure_ascii=False)
    print(f"[{dom}] 有效节点 {N} 张")

    # ---- 后代 ----
    img_dir = os.path.join(wd, "images")
    os.makedirs(img_dir, exist_ok=True)
    desc = []
    if dom == "koniq":
        src_positions = list(range(0, N, max(1, N // n_desc_src)))[:n_desc_src]
        for k, pos in enumerate(src_positions):
            for f, fam in enumerate(desc_fams):
                node = N + 2 * k + f
                path = os.path.join(img_dir, f"{node:04d}_{fam}.jpg")
                if not os.path.exists(path):
                    gen_desc(nodes[pos]["path"], fam, path)
                desc.append({"node": node, "src_node": pos, "family": fam, "path": path})
    else:
        # SPAQ 后代：只从"梯子源"节点里选，直接复用 runs/ladder2/images 的 blur_L2/jpeg_L2
        d2 = os.path.join(cfg.runs_dir, "ladder2")
        l2_manifest = json.load(open(os.path.join(d2, "manifest.json")))
        lad_nodes = [n_ for n_ in nodes if "ladder_src" in n_][:n_desc_src]
        for k, n_ in enumerate(lad_nodes):
            for f, fam in enumerate(desc_fams):
                file = [m["file"] for m in l2_manifest
                        if m["domain"] == "spaq" and m["src_idx"] == n_["ladder_src"]
                        and m["family"] == fam and m["level"] == 2][0]
                desc.append({"node": N + 2 * k + f, "src_node": n_["node"], "family": fam,
                             "path": os.path.join(d2, "images", file)})
    print(f"[{dom}] 后代 {len(desc)} 张")

    # ---- 特征（ocv_hard 对用）----
    feats = {}
    if n_ocv:
        fpath = os.path.join(wd, f"features_{dom}.json")
        if os.path.exists(fpath):
            feats = json.load(open(fpath))
        else:
            for n_ in nodes:
                feats[str(n_["node"])] = opencv_features(n_["path"])
            json.dump(feats, open(fpath, "w"))

    # ---- 配对 ----
    rng = random.Random(cfg.seed)
    order = sorted(range(N), key=lambda i: nodes[i]["old"])
    pairs = []
    used = set()

    def add(i, j, kind, orders=1):
        pairs.append({"i": i, "j": j, "kind": kind, "orders": orders})

    for r in range(N - 1):
        a, b = order[r], order[r + 1]
        used.add((min(a, b), max(a, b)))
        add(a, b, "adjacent", orders=adj_orders)
    got = 0
    while got < n_random:
        a, b = rng.sample(range(N), 2)
        key = (min(a, b), max(a, b))
        if key not in used:
            used.add(key)
            add(a, b, "random")
            got += 1
    if n_spread:
        iqr = []
        for n_ in nodes:
            ss = [v for v in n_["skill_scores"].values() if v is not None]
            q = np.percentile(ss, [25, 75]) if len(ss) >= 2 else [0, 0]
            iqr.append(float(q[1] - q[0]))
        cand = [(i, j) for i in range(N) for j in range(i + 1, N)
                if abs(nodes[i]["old"] - nodes[j]["old"]) <= 0.15 and (i, j) not in used]
        cand.sort(key=lambda t: -(iqr[t[0]] + iqr[t[1]]))
        for a, b in cand[:n_spread]:
            used.add((a, b))
            add(a, b, "spread")
    if n_ocv:
        laps = np.array([feats[str(i)]["lap_var"] for i in range(N)])
        m_, s_ = laps.mean(), laps.std() + 1e-9
        cand2 = [(i, j) for i in range(N) for j in range(i + 1, N)
                 if abs(nodes[i]["old"] - nodes[j]["old"]) <= 0.15 and (i, j) not in used]
        cand2.sort(key=lambda t: -abs((laps[t[0]] - m_) / s_ - (laps[t[1]] - m_) / s_))
        for a, b in cand2[:n_ocv]:
            used.add((a, b))
            add(a, b, "ocv_hard")
    rank_of = {idx: r for r, idx in enumerate(order)}
    for d in desc:
        r = rank_of[d["src_node"]]
        for dr in (1, -1, 2):
            rr = min(max(r + dr, 0), N - 1)
            if rr != r:
                add(d["node"], order[rr], "desc_nat")
    for d in desc:
        if d["family"] == desc_fams[0] and (d["node"] - N) // 2 < 24:
            add(d["src_node"], d["node"], "probe", orders=2)
    n_calls = sum(p["orders"] for p in pairs)
    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + p["orders"]
    print(f"[{dom}] 配对 {len(pairs)} 组 → {n_calls} 次调用 {kinds}")
    json.dump(pairs, open(os.path.join(wd, f"pairs_{dom}.json"), "w"))
    if args.dry_run:
        print("[dry-run] 到此为止，未碰新 API。")
        return

    # ---- 对决 ----
    node_paths = {n_["node"]: n_["path"] for n_ in nodes}
    for d in desc:
        node_paths[d["node"]] = d["path"]
    prompt = build_pairwise_prompt()
    duel_jobs = []
    for p in pairs:
        for o in range(p["orders"]):
            first, second = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
            duel_jobs.append({"i": p["i"], "j": p["j"], "first": first, "second": second, "kind": p["kind"]})

    async def duel(job):
        text, _ = await client.compare_images(node_paths[job["first"]], node_paths[job["second"]], prompt)
        w = _parse_winner(text)
        if w == "A":
            job["winner_node"], job["loser_node"] = job["first"], job["second"]
        elif w == "B":
            job["winner_node"], job["loser_node"] = job["second"], job["first"]
        else:
            job["winner_node"] = job["loser_node"] = None
        return job

    dpath = os.path.join(wd, f"duels_{dom}.json")
    duels = json.load(open(dpath, encoding="utf-8")) if os.path.exists(dpath) else []
    done_keys = {(d["i"], d["j"], d["first"]) for d in duels}
    todo = [j for j in duel_jobs if (j["i"], j["j"], j["first"]) not in done_keys]
    print(f"[{dom}] 对决 {len(todo)} 场（已完成 {len(done_keys)}）")
    results = await gather_with_progress([duel(j) for j in todo], every=500, label=f"{dom}-duels")
    n_bad = sum(1 for r in results if isinstance(r, Exception) or r["winner_node"] is None)
    duels.extend([r for r in results if not isinstance(r, Exception)])
    json.dump(duels, open(dpath, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[{dom}] 完成，无效 {n_bad} 场；账本 {client.ledger()}")

    # ---- 后代旧分（仅 KonIQ，供 G2b 诊断）----
    desc_scores = {}
    if dom == "koniq":
        dsp = os.path.join(wd, "desc_scores_koniq.json")
        desc_scores = json.load(open(dsp)) if os.path.exists(dsp) else {}
        todo_desc = [d for d in desc if str(d["node"]) not in desc_scores]
        if todo_desc:
            refs = [SimpleNamespace(img_id=str(d["node"]), path=d["path"], dataset="koniq") for d in todo_desc]
            rrows = await run_r2(client, refs, "koniq", cfg.scales["koniq"], dynamic=False, rules_by_skill=None)
            for d, r in zip(todo_desc, rrows):
                desc_scores[str(d["node"])] = r["score"]
            json.dump(desc_scores, open(dsp, "w"))
            print(f"[koniq] 后代旧分 {len(todo_desc)} 张；账本 {client.ledger()}")

    # ---- 门控 ----
    n_nodes = N + len(desc)
    probe = [d for d in duels if d["kind"] == "probe" and d["winner_node"] is not None]
    g1 = sum(1 for d in probe if d["winner_node"] == d["i"]) / len(probe) if probe else 0.0
    old_scores = [n_["old"] for n_ in nodes]

    nn = [d for d in duels if d["winner_node"] is not None and d["i"] < N and d["j"] < N and d["kind"] != "probe"]
    random.Random(cfg.seed + 1).shuffle(nn)
    folds = [nn[k::5] for k in range(5)]
    bt_hits = old_hits = tot = 0
    for k in range(5):
        hold_keys = {(d["i"], d["j"], d["winner_node"]) for d in folds[k]}
        train = [d for d in duels if (d["i"], d["j"], d["winner_node"]) not in hold_keys]
        lat = fit_bt(build_wins(train, desc, n_nodes))
        for d in folds[k]:
            bt_hits += (lat[d["i"]] > lat[d["j"]]) == (d["winner_node"] == d["i"])
            old_hits += (old_scores[d["i"]] > old_scores[d["j"]]) == (d["winner_node"] == d["i"])
            tot += 1
    g2_bt, g2_old = bt_hits / tot, old_hits / tot

    judged = [d for d in duels if d["winner_node"] is not None]
    random.Random(cfg.seed).shuffle(judged)
    lat_a = fit_bt(build_wins(judged[: len(judged) // 2], desc, n_nodes))
    lat_b = fit_bt(build_wins(judged[len(judged) // 2:], desc, n_nodes))
    g3 = float(spearmanr(lat_a[:N], lat_b[:N]).statistic)
    full_lat = fit_bt(build_wins(duels, desc, n_nodes))
    echo = float(spearmanr(full_lat[:N], old_scores).statistic)

    report = {
        "domain": dom, "n_nodes": N, "n_desc": len(desc),
        "G1_probe_acc": round(float(g1), 4), "G1_pass": bool(g1 >= 0.90),
        "G2_bt_holdout_acc": round(float(g2_bt), 4), "G2_old_holdout_acc": round(float(g2_old), 4),
        "G2_pass": bool(g2_bt > g2_old + 0.02),
        "G3_split_half_srcc": round(float(g3), 4), "G3_pass": bool(g3 >= 0.75),
        "echo_srcc_bt_vs_old": round(float(echo), 4),
        "n_duels": len(duels), "n_invalid": sum(1 for d in duels if d["winner_node"] is None),
        "ledger": client.ledger(),
    }
    report["verdict"] = "GO" if (report["G1_pass"] and report["G2_pass"] and report["G3_pass"]) else "CHECK"
    json.dump(report, open(os.path.join(wd, f"report_{dom}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # BT 排行榜落盘（Router v2 主监督信号）
    ranking = sorted(({"node": i, "path": nodes[i]["path"], "bt": float(full_lat[i]),
                       "old": old_scores[i]} for i in range(N)), key=lambda r: -r["bt"])
    json.dump(ranking, open(os.path.join(wd, f"ranking_{dom}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["koniq", "spaq"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
