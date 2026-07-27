# -*- coding: utf-8 -*-
"""V3-D1：KonIQ BT 锦标赛扩容（999 → 2,500 节点）。

合规（ADR-0003）：仅 Train 像素，零 MOS。老节点/对决/后代全部复用（缓存），
只新增：1,501 张 Train 图的 3 专家分（瘦身池 TECH/GLOBAL/CONTENT）+ ~7,800 场新对决。

边配比（F-013 密度定律 + 反回声混合）：
- 新节点相邻对 ×2 序（局部微调）；
- 随机长程对（新老混合，全局骨架）；
- 专家分歧对 + OpenCV 难分对（破回声）。
扩容后按同一门控复核（G2 留出对决 / G3 分裂半），写 ranking_koniq_v3.json。

用法： python scripts/92_expand_tournament.py [--dry-run]
"""
import argparse
import asyncio
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
from iqa_agent.prompts.pairwise import build_pairwise_prompt
from iqa_agent.prompts.skills import build_skill_prompt
from iqa_agent.scoring import parse_score

NEW_N = 1501
POOL3 = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
KNOWN_W = 2.0


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


def opencv_features(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    c = gray[1:-1, 1:-1]
    lap = -4 * c + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(lap))


def fit_bt(wins):
    from scipy.optimize import minimize
    n = wins.shape[0]
    idx_i, idx_j = np.nonzero(wins)
    w_ij, g_ij = wins[idx_i, idx_j], wins[idx_i, idx_j] + wins[idx_j, idx_i]

    def nll_grad(theta):
        la = np.logaddexp(theta[idx_i], theta[idx_j])
        p = np.exp(theta[idx_i] - la)
        ll = -float(np.sum(w_ij * theta[idx_i])) + float(np.sum(g_ij * la)) + 1e-6 * float(np.sum(theta ** 2))
        grad = np.zeros(n)
        np.add.at(grad, idx_i, -w_ij + g_ij * p)
        np.add.at(grad, idx_j, g_ij * (1 - p))
        return ll, grad + 2e-6 * theta

    res = minimize(nll_grad, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": 5000})
    return res.x


async def main_async(args):
    cfg = get_config()
    wd = os.path.join(cfg.runs_dir, "full_tournament")
    client = VLMClient(cfg, cfg.model_main)

    # ---- 老节点（999，5 专家分缓存免费）----
    ws = jload(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"))
    old_nodes = [{"path": r["path"], "old": r["score"],
                  "skill_scores": json.loads(r["skill_scores"]) if isinstance(r["skill_scores"], str) else r["skill_scores"]}
                 for r in ws if r.get("score") is not None]
    old_paths = {n["path"] for n in old_nodes}
    print(f"老节点 {len(old_nodes)}")

    # ---- 新节点（1,501，3 专家瘦身池评分）----
    all_train = load_images(cfg, "koniq_train")
    cand = [r for r in all_train if r.path not in old_paths]
    rng = random.Random(cfg.seed + 77)
    rng.shuffle(cand)
    new_refs = cand[:NEW_N]
    ns_path = os.path.join(wd, "new_node_scores.json")
    new_scores = jload(ns_path) if os.path.exists(ns_path) else {}

    async def score_one(ref, sk):
        text, _ = await client.score_image(ref.path, build_skill_prompt(sk, "koniq", rules=None), temperature=0.0)
        p = parse_score(text, cfg.scales["koniq"])
        return ref.path, sk, (p["score"] if p else None)

    jobs = [score_one(r, sk) for r in new_refs for sk in POOL3
            if new_scores.get(r.path, {}).get(sk) is None]
    print(f"新节点 {len(new_refs)}，3 专家待评 {len(jobs)} 次")
    if not args.dry_run and jobs:
        rows = await gather_with_progress(jobs, every=500, label="new-node-score")
        for r in rows:
            if not isinstance(r, Exception) and r[2] is not None:
                new_scores.setdefault(r[0], {})[r[1]] = r[2]
        with open(ns_path, "w", encoding="utf-8") as f:
            json.dump(new_scores, f)
        print(f"新节点评分落盘；账本 {client.ledger()}")

    nodes = old_nodes[:]
    for r in new_refs:
        ss = new_scores.get(r.path, {})
        if all(ss.get(sk) is not None for sk in POOL3):
            nodes.append({"path": r.path, "old": float(np.mean([ss[sk] for sk in POOL3])),
                          "skill_scores": ss, "new": True})
    for i, n in enumerate(nodes):
        n["node"] = i
    N = len(nodes)
    print(f"总节点 {N}")
    if args.dry_run:
        print("[dry-run] 配对预估见下，未碰对决。")

    # ---- OpenCV 特征（新节点，本地）----
    fpath = os.path.join(wd, "features_koniq_v3.json")
    feats = jload(fpath) if os.path.exists(fpath) else {}
    for n in nodes:
        if n.get("new") and str(n["node"]) not in feats:
            feats[str(n["node"])] = opencv_features(n["path"])
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(feats, f)

    # ---- 配对 ----
    rng2 = random.Random(cfg.seed + 78)
    order = sorted(range(N), key=lambda i: nodes[i]["old"])
    new_idx = [n["node"] for n in nodes if n.get("new")]
    pairs, used = [], set()

    def add(i, j, kind, orders=1):
        pairs.append({"i": i, "j": j, "kind": kind, "orders": orders})

    pos = {idx: r for r, idx in enumerate(order)}
    for ni in new_idx:  # 新节点相邻 ×2 序
        r = pos[ni]
        for dr in (1, -1):
            rr = r + dr
            if 0 <= rr < N and order[rr] != ni:
                a, b = sorted((ni, order[rr]))
                used.add((a, b))
                add(a, b, "adjacent", orders=2)
    got = 0
    while got < 14000:  # 随机长程（新老混合）；G3 密度补强至两半 ~6.5 边/节点（F-013）
        a, b = rng2.sample(range(N), 2)
        key = (min(a, b), max(a, b))
        if key not in used:
            used.add(key)
            add(a, b, "random")
            got += 1
    iqr = []
    for n in nodes:
        ss = [v for v in n["skill_scores"].values() if v is not None]
        q = np.percentile(ss, [25, 75]) if len(ss) >= 2 else [0, 0]
        iqr.append(float(q[1] - q[0]))
    cand_sp = [(i, j) for i in new_idx for j in range(N)
               if i != j and abs(nodes[i]["old"] - nodes[j]["old"]) <= 0.15
               and (min(i, j), max(i, j)) not in used]
    cand_sp.sort(key=lambda t: -(iqr[t[0]] + iqr[t[1]]))
    for a, b in cand_sp[:400]:
        used.add((min(a, b), max(a, b)))
        add(a, b, "spread")
    laps = np.array([feats.get(str(i), 0.0) for i in range(N)])
    m_, s_ = laps.mean(), laps.std() + 1e-9
    cand_oc = [(i, j) for i in new_idx for j in range(N)
               if i != j and abs(nodes[i]["old"] - nodes[j]["old"]) <= 0.15
               and (min(i, j), max(i, j)) not in used]
    cand_oc.sort(key=lambda t: -abs((laps[t[0]] - m_) / s_ - (laps[t[1]] - m_) / s_))
    for a, b in cand_oc[:300]:
        used.add((min(a, b), max(a, b)))
        add(a, b, "ocv_hard")
    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + p["orders"]
    print(f"新配对 {len(pairs)} 组 → {sum(p['orders'] for p in pairs)} 场 {kinds}")
    if args.dry_run:
        return

    # ---- 对决（增量，断点续跑）----
    # 后代重编号：旧 duels 中后代 id 为 999..1318（86 时 N=999），
    # 与新自然节点 999..2499 冲突 → 统一平移到 N..N+319（N=新自然节点数）
    N_SHIFT = N - 999
    node_paths = {n["node"]: n["path"] for n in nodes}
    dpath = os.path.join(wd, "duels_koniq.json")
    duels = jload(dpath)
    for d in duels:  # 旧对决的后代 id 平移（一次性，幂等由 shifted 标记保证）
        if d.pop("v3_shifted", None):
            continue
        for key in ("i", "j", "first", "second", "winner_node", "loser_node"):
            if d.get(key) is not None and d[key] >= 999:
                d[key] += N_SHIFT
        d["v3_shifted"] = True
    # 后代路径表（86 images 目录，文件名节点号 999..1318 → N..N+319）
    img_dir = os.path.join(wd, "images")
    if os.path.isdir(img_dir):
        for fn in os.listdir(img_dir):
            node_id = int(fn.split("_")[0])
            node_paths[node_id + N_SHIFT] = os.path.join(img_dir, fn)
    done_keys = {(d["i"], d["j"], d["first"]) for d in duels}
    prompt = build_pairwise_prompt()
    duel_jobs = []
    for p in pairs:
        for o in range(p["orders"]):
            first, second = (p["i"], p["j"]) if o == 0 else (p["j"], p["i"])
            if (p["i"], p["j"], first) not in done_keys:
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

    print(f"新对决 {len(duel_jobs)} 场（已完成 {len(done_keys)}）")
    for chunk_i in range(0, len(duel_jobs), 1000):
        chunk = duel_jobs[chunk_i: chunk_i + 1000]
        part = await gather_with_progress([duel(j) for j in chunk], every=250, label=f"v3-duels[{chunk_i // 1000}]")
        errs = [r for r in part if isinstance(r, Exception)]
        if errs:
            print(f"  ⚠️ 本块 {len(errs)} 个失败，首例: {type(errs[0]).__name__}: {str(errs[0])[:200]}")
        duels.extend([r for r in part if not isinstance(r, Exception)])
        with open(dpath, "w", encoding="utf-8") as f:  # 增量落盘（断电教训）
            json.dump(duels, f, ensure_ascii=False)
    print(f"对决完成；账本 {client.ledger()}")

    # ---- BT + 门控 ----
    n_nodes = max(max(d["i"], d["j"]) for d in duels if d.get("winner_node") is not None) + 1
    # 已知边：86 的确定性后代规则重建（src_positions=range(0,999,6)[:160]，
    # 后代节点平移后为 N+2k(blur)/N+2k+1(noise)，src>desc 权重 KNOWN_W）
    n_old = len(old_nodes)
    src_positions = list(range(0, n_old, max(1, n_old // 160)))[:160]
    desc_edges = [{"src_node": pos, "node": N + 2 * k + f}
                  for k, pos in enumerate(src_positions) for f in (0, 1)]

    def build_wins(dd, holdout=None):
        wins = np.zeros((n_nodes, n_nodes))
        for d in dd:
            if d.get("winner_node") is not None:
                wins[d["winner_node"], d["loser_node"]] += 1
        for e in desc_edges:
            if holdout and (e["src_node"], e["node"]) in holdout:
                continue
            wins[e["src_node"], e["node"]] += KNOWN_W
        return wins

    old_scores = {i: nodes[i]["old"] for i in range(N)}
    nn = [d for d in duels if d["winner_node"] is not None and d["i"] < N and d["j"] < N and d["kind"] != "probe"]
    random.Random(cfg.seed + 1).shuffle(nn)
    folds = [nn[k::5] for k in range(5)]
    bt_hits = old_hits = tot = 0
    for k in range(5):
        hold_keys = {(d["i"], d["j"], d["winner_node"]) for d in folds[k]}
        train = [d for d in duels if (d["i"], d["j"], d["winner_node"]) not in hold_keys]
        lat = fit_bt(build_wins(train))
        for d in folds[k]:
            bt_hits += (lat[d["i"]] > lat[d["j"]]) == (d["winner_node"] == d["i"])
            old_hits += (old_scores[d["i"]] > old_scores[d["j"]]) == (d["winner_node"] == d["i"])
            tot += 1
    judged = [d for d in duels if d["winner_node"] is not None]
    random.Random(cfg.seed).shuffle(judged)
    lat_a = fit_bt(build_wins(judged[: len(judged) // 2]))
    lat_b = fit_bt(build_wins(judged[len(judged) // 2:]))
    g3 = float(spearmanr(lat_a[:N], lat_b[:N]).statistic)
    full_lat = fit_bt(build_wins(duels))
    report = {
        "n_nodes": N, "n_new": len(new_idx), "n_duels": len(duels),
        "G2_bt_holdout_acc": round(bt_hits / tot, 4), "G2_old_holdout_acc": round(old_hits / tot, 4),
        "G2_pass": bool(bt_hits / tot > old_hits / tot + 0.02),
        "G3_split_half_srcc": round(g3, 4), "G3_pass": bool(g3 >= 0.75),
        "ledger": client.ledger(),
    }
    report["verdict"] = "GO" if (report["G2_pass"] and report["G3_pass"]) else "CHECK"
    with open(os.path.join(wd, "report_koniq_v3.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    ranking = sorted(({"node": i, "path": nodes[i]["path"], "bt": float(full_lat[i]),
                       "old": old_scores[i]} for i in range(N)), key=lambda r: -r["bt"])
    with open(os.path.join(wd, "ranking_koniq_v3.json"), "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
