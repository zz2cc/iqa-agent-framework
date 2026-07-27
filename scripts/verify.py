# -*- coding: utf-8 -*-
r"""
复现验证脚本（唯一入口）。从 GitHub clone 后只需配好 .env 和数据集即可运行。

用法:
  python scripts/verify.py                 零 API，缓存主表 + R6 管线(缓存专家分)
  python scripts/verify.py --api           同上 + KonIQ×200 + SPAQ×200 API 重调专家分
  python scripts/verify.py --html          同上(零API) + 输出自包含 HTML 报告
  python scripts/verify.py --api --html    全部

输出 Section:
  [1/5] 环境自检
  [2/5] 主表验证 (R1-bare ~ R3, R1-anchor v3, 8B R1-bare) — 纯缓存
  [3/5] R6 统一框架管线
  [4/5] CKE 规则展示
  [5/5] (--api 时) API 抽样重跑 or (--html 时) HTML 报告
"""
import csv, json, os, sys, time, argparse, random
import numpy as np
from PIL import Image

# 编码安全: Windows GBK tolerate
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iqa_agent.config import get_config
from iqa_agent.router import opencv_features
from iqa_agent.metrics import compute_metrics
from iqa_agent.data import load_mos

# ═══════════════ 工具函数 ═══════════════

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def read_scores_csv(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[r["img_id"]] = r
    return rows

def jload(p):
    if not os.path.exists(p):
        return {}
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return {}

def load_pred(path):
    """读 scores.csv 返回 {img_id: float}。"""
    pred = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            s = row.get("score", "")
            if s and s.strip():
                pred[row["img_id"]] = float(s)
    return pred

def load_expert_scores(r2_path, pool):
    """从 R2 scores.csv 提取专家分 {img_id: {SKILL: float}}。"""
    expert = {}
    with open(r2_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ss_raw = r.get("skill_scores", "")
            if not ss_raw:
                continue
            try:
                ss = json.loads(ss_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            entry = {}
            for sk in pool:
                v = ss.get(sk)
                if v is not None:
                    entry[sk] = float(v)
            if len(entry) == len(pool):
                expert[r["img_id"]] = entry
    return expert

def r6_bare_vote(iid, r1b, paras):
    """裸问释义投票：原句 + 3 条同义问法取均值。"""
    vals = [r1b.get(iid)]
    if paras and iid in paras:
        for k in ("1", "2", "3"):
            v = paras[iid].get(k)
            if v is not None:
                vals.append(float(v))
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

# ═══════════════ 主逻辑 ═══════════════

def main():
    ap = argparse.ArgumentParser(description="IQA Agent 复现验证")
    ap.add_argument("--api", action="store_true", help="KonIQ+SPAQ 各200张 API 重跑专家分")
    ap.add_argument("--html", action="store_true", help="输出自包含 HTML 报告")
    ap.add_argument("--limit", type=int, default=None, help="限制 R6 管线数量(调试用)")
    args = ap.parse_args()

    cfg = get_config()
    POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
    FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]

    # ═══════════════ [1/5] 环境自检 ═══════════════
    print("=" * 72)
    print("[1/5] 环境自检")
    print("=" * 72)
    checks = []
    # Python
    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    # Packages
    for mod, name in [("numpy", "numpy"), ("scipy.stats", "scipy"), ("PIL", "pillow")]:
        try:
            __import__(mod)
            checks.append((name, True, "OK"))
        except ImportError:
            checks.append((name, False, "MISSING — pip install " + name))
    # openai
    try:
        __import__("openai")
        checks.append(("openai", True, "OK"))
    except ImportError:
        checks.append(("openai", False, "MISSING — pip install openai (仅 --api 需要)"))

    # Datasets
    for lab, d in [("KonIQ", cfg.koniq_img_dir), ("SPAQ", cfg.spaq_img_dir)]:
        ok = os.path.isdir(d) and len(os.listdir(d)) > 100
        checks.append((f"数据集 {lab}", ok, d if ok else f"MISSING — {d}"))

    # .env
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    checks.append(("API Key (.env)", has_key, "OK" if has_key else "MISSING — cp .env.example .env"))

    # Frozen files
    for lab, p in [("W 矩阵", "runs/router_v3/fusion_koniq.json"),
                   ("R2 KonIQ 专家", "runs/final/r2_koniq/scores.csv"),
                   ("R2 SPAQ 专家", "runs/final/r2_spaq/scores.csv")]:
        ok = os.path.exists(os.path.join(cfg.runs_dir, *p.split("/")[1:]))
        checks.append((lab, ok, "OK" if ok else "MISSING"))

    all_ok = True
    for name, ok, detail in checks:
        mark = "V" if ok else "X"
        print(f"  [{mark}] {name:<20s}  {detail}")
        if not ok and "MISSING" not in str(detail) and "pip" not in str(detail):
            all_ok = False

    if not all_ok:
        print("\n  *** 环境自检未通过，请修复上述问题后重试 ***\n")

    # ═══════════════ [2/5] 主表验证 ═══════════════
    print("\n" + "=" * 72)
    print("[2/5] 主表验证 (纯缓存，零 API)")
    print("=" * 72)

    ARMS = [
        ("R1-bare",         "r1b_koniq",         "r1b_spaq",         "一句话裸问"),
        ("R1-rich",         "r1r_koniq",         "r1r_spaq",         "单专家+完整细则"),
        ("R2",              "r2_koniq",          "r2_spaq",          "5专家+截尾融合"),
        ("R2.5",            "r25_koniq",         "r25_spaq",         "+动态分诊Router"),
        ("R3",              "r3_koniq",          "r3_spaq",          "+CKE自进化规则库"),
        ("R6 (统一框架)",    "r6_koniq",          "r6_spaq",          "门控融合+投票混合"),
        ("R1-anchor v3",    "r1anchor_v3_koniq", "r1anchor_v3_spaq","轻人设+锚点+程序"),
        ("8B R1-bare",      "r1b_8b_koniq",      "r1b_8b_spaq",     "8B模型裸问"),
    ]

    fmt_hdr = "  {:<20s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>6s}"
    fmt_row = "  {:<20s}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>6s}"

    print(f"\n  {'─'*62}")
    print(fmt_hdr.format("臂", "KonIQ_SRCC", "KonIQ_MAE", "SPAQ_SRCC", "SPAQ_MAE", "来源"))
    print(f"  {'─'*62}")

    table_data = []  # for HTML
    for arm_name, d_k, d_s, desc in ARMS:
        srcc_k = mae_k = srcc_s = mae_s = None
        src = ""

        # KonIQ
        p_k = os.path.join(cfg.runs_dir, "final" if not arm_name.startswith(("R1-anchor","8B")) else "posthoc", d_k, "scores.csv")
        if os.path.exists(p_k):
            pred_k = load_pred(p_k)
            mos_k = load_mos(cfg, "koniq_val")
            m = compute_metrics(pred_k, mos_k)
            srcc_k, mae_k = m["SRCC"], m["MAE"]
            src = "缓存"
        else:
            src = "缺失"

        # SPAQ
        p_s = os.path.join(cfg.runs_dir, "final" if not arm_name.startswith(("R1-anchor","8B")) else "posthoc", d_s, "scores.csv")
        if os.path.exists(p_s):
            pred_s = load_pred(p_s)
            mos_s = load_mos(cfg, "spaq_test")
            m = compute_metrics(pred_s, mos_s)
            srcc_s, mae_s = m["SRCC"], m["MAE"]

        if srcc_k is not None:
            print(fmt_row.format(arm_name, srcc_k, mae_k, srcc_s, mae_s, src))
        else:
            print(f"  {arm_name:<20s}  {'MISSING':>8s}")

        table_data.append((arm_name, srcc_k, mae_k, srcc_s, mae_s, src, desc))

    print(f"  {'─'*62}")
    print(f"\n  共计 {sum(1 for t in table_data if t[1] is not None)} 臂数据完整。")

    # ═══════════════ [3/5] R6 管线 ═══════════════
    print("\n" + "=" * 72)
    print("[3/5] R6 统一框架管线 (缓存专家分，零 API)")
    print("=" * 72)

    fk = jload(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"))
    W = np.array(fk["W"]); mu = np.array(fk["mu"])
    sd = np.array([s if s > 1e-6 else 1.0 for s in fk["sd"]])

    for ds, eval_ds, alpha, r2_dir, r1b_dir, para_file, r6_cache_dir, gate_desc in [
        ("koniq", "koniq_val", 0.6, "r2_koniq", "r1b_koniq", "r6_koniq_paras.json",
         "r6_koniq", "W 矩阵逐图动态门控"),
        ("spaq", "spaq_test", 0.3, "r2_spaq", "r1b_spaq", "r6_spaq_paras.json",
         "r6_spaq", "SPAQ 域门控(缺失→等权回退)"),
    ]:
        print(f"\n  ── {ds.upper()} R6 管线 (α={alpha}, {gate_desc}) ──")

        r2_path = os.path.join(cfg.runs_dir, "final", r2_dir, "scores.csv")
        r1b_path = os.path.join(cfg.runs_dir, "final", r1b_dir, "scores.csv")

        if not os.path.exists(r2_path):
            print(f"    [SKIP] {r2_path} 不存在")
            continue

        expert = load_expert_scores(r2_path, POOL)
        r1b = load_pred(r1b_path)
        paras = jload(os.path.join(cfg.runs_dir, para_file))
        mos = load_mos(cfg, eval_ds)
        img_dir = cfg.koniq_img_dir if ds == "koniq" else cfg.spaq_img_dir

        ids_all = sorted(set(expert.keys()) & set(r1b.keys()) & set(mos.keys()))
        if args.limit:
            random.seed(42)
            ids_all = sorted(random.sample(ids_all, min(args.limit, len(ids_all))))
        print(f"    图像数: {len(ids_all)}")

        # Load cached R6 for comparison
        cached_path = os.path.join(cfg.runs_dir, "final", r6_cache_dir, "scores.csv")
        cached = load_pred(cached_path) if os.path.exists(cached_path) else {}
        # Also try unified posthoc for SPAQ
        if ds == "spaq":
            cached_u_path = os.path.join(cfg.runs_dir, "posthoc", "r6_unified_spaq", "scores.csv")
            if os.path.exists(cached_u_path):
                cached = load_pred(cached_u_path)
                print(f"    对比目标: runs/posthoc/r6_unified_spaq (统一版, 等权)")

        rows, n_skip = [], 0
        t0 = time.time()
        show_detail = True  # 打前 3 张图的完整流程
        detail_count = 0

        for idx, iid in enumerate(ids_all):
            es = expert[iid]
            s3 = [es[sk] for sk in POOL]
            sp = float(np.std(s3))

            # OpenCV 特征
            fp = os.path.join(img_dir, iid)
            if not os.path.exists(fp):
                n_skip += 1; continue
            img = Image.open(fp)
            if ds == "spaq":
                img.thumbnail((1568, 1568), Image.BICUBIC)
            f_raw = opencv_features(img)
            f = np.array([f_raw["lap_var"], f_raw["noise"], f_raw["colorful"],
                          f_raw["bright"], f_raw["logpix"], f_raw["aspect"], sp], dtype=float)
            for j, k in enumerate(FEAT_KEYS):
                if k in ("lap_var", "noise", "colorful"):
                    f[j] = np.log(max(f[j], 1e-6))

            # 门控
            if ds == "koniq":
                g = softmax(W @ ((f - mu) / sd))
                fus = float(g @ np.array(s3))
            else:
                # SPAQ: 等权回退
                g = np.array([1/3, 1/3, 1/3])
                fus = float(np.mean(s3))

            # 投票分
            vote = r6_bare_vote(iid, r1b, paras)
            if vote is None:
                n_skip += 1; continue

            final = round(alpha * fus + (1 - alpha) * vote, 4)
            rows.append({"img_id": iid, "score": final})

            # 打印详细流程（前 3 张）
            if show_detail and detail_count < 3:
                print(f"\n    [{detail_count+1}/3] 图片 {iid}")
                print(f"      专家分 (缓存): TECH={s3[0]:.2f}  GLOBAL={s3[1]:.2f}  CONTENT={s3[2]:.2f}  spread={sp:.2f}")
                print(f"      像素特征:      lap_var={f_raw['lap_var']:.1f}  noise={f_raw['noise']:.3f}  colorful={f_raw['colorful']:.1f}")
                print(f"                      bright={f_raw['bright']:.1f}  logpix={f_raw['logpix']:.2f}  aspect={f_raw['aspect']:.3f}")
                if ds == "koniq":
                    logits = W @ ((f - mu) / sd)
                    print(f"      标准化后×W:    logits=[{logits[0]:+.3f}, {logits[1]:+.3f}, {logits[2]:+.3f}]")
                print(f"      话语权 g:      TECH={g[0]:.3f}  GLOBAL={g[1]:.3f}  CONTENT={g[2]:.3f}")
                print(f"      融合分 s_fus:  {fus:.4f}")
                print(f"      投票分 s_vote: {vote:.4f}")
                print(f"      最终分:        {alpha}*{fus:.4f} + {1-alpha}*{vote:.4f} = {final:.4f}")
                if iid in cached:
                    print(f"      已缓存 R6 分:  {cached[iid]:.4f}")
                detail_count += 1

            # 每 200 张汇报
            if (idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                speed = (idx + 1) / elapsed
                print(f"    [{idx+1}/{len(ids_all)}] {elapsed:.0f}s ({speed:.1f} 张/s)", flush=True)

        print(f"    耗时: {time.time()-t0:.0f}s  跳过: {n_skip}")

        # 对比
        pred = {r["img_id"]: r["score"] for r in rows}
        m = compute_metrics(pred, mos)
        print(f"\n    管线重算:  SRCC={m['SRCC']:.4f}  MAE={m['MAE']:.4f}  n={m['n']}")

        if cached:
            common = sorted(set(pred.keys()) & set(cached.keys()))
            if common:
                cp = {iid: cached[iid] for iid in common}
                pp = {iid: pred[iid] for iid in common}
                cm = compute_metrics(cp, mos)
                pm = compute_metrics(pp, mos)
                d_srcc = pm["SRCC"] - cm["SRCC"]
                d_mae = pm["MAE"] - cm["MAE"]
                print(f"    已缓存 R6:  SRCC={cm['SRCC']:.4f}  MAE={cm['MAE']:.4f}  n={cm['n']}")
                print(f"    差值:       SRCC {'+' if d_srcc>0 else ''}{d_srcc:.4f}  MAE {d_mae:+.4f}")
                if abs(d_srcc) < 0.01 and abs(d_mae) < 0.01:
                    print(f"    ✓ 管线重算与缓存一致")
                else:
                    print(f"    ! 差异 > 0.01，请检查")

    # ═══════════════ [4/5] CKE 规则 ═══════════════
    print("\n" + "=" * 72)
    print("[4/5] CKE 自进化规则库 (已被最终框架移除)")
    print("=" * 72)

    cke_lib = os.path.join(cfg.runs_dir, "cke", "final_library.json")
    if os.path.exists(cke_lib):
        rules = jload(cke_lib)
        if isinstance(rules, list):
            print(f"\n  共 {len(rules)} 条规则，经 CKE 四轮迭代 + 双门控筛选。")
            print("  内部指标: 阶梯单调性 0.455→0.474 (+0.019) / B-C分歧 0.402→0.330 (−0.07)")
            print("  外部结果: KonIQ SRCC 仅 +0.001 — 自洽≠对齐。\n")
            for i, r in enumerate(rules, 1):
                if isinstance(r, dict):
                    print(f"  规则 {i}: {r.get('rule', r.get('text', str(r)))[:120]}")
                else:
                    print(f"  规则 {i}: {str(r)[:120]}")
        else:
            print(f"\n  规则库内容: {str(rules)[:500]}")
    else:
        print(f"\n  [注意] CKE 规则库文件不存在: {cke_lib}")

    # ═══════════════ [5/5] API 抽样 ═══════════════
    if args.api:
        print("\n" + "=" * 72)
        print("[5/5] API 抽样重跑 (KonIQ×200 + SPAQ×200，32B 专家分)")
        print("=" * 72)

        if not has_key:
            print("  [SKIP] 未配置 API Key，跳过")
        else:
            from iqa_agent.client import VLMClient, gather_with_progress
            from iqa_agent.prompts.skills import build_skill_prompt, BARE_PARAS
            from iqa_agent.scoring import parse_score
            import asyncio

            async def api_verify():
                client = VLMClient(cfg, cfg.model_main)  # 32B

                for ds, eval_ds, lo, hi, r2_dir, alpha, gate_desc in [
                    ("koniq", "koniq_val", 1, 5, "r2_koniq", 0.6, "W矩阵动态门控"),
                    ("spaq", "spaq_test", 0, 10, "r2_spaq", 0.3, "等权回退(缺门控文件)"),
                ]:
                    print(f"\n  ── {ds.upper()} API 抽样 ──")
                    img_dir = cfg.koniq_img_dir if ds == "koniq" else cfg.spaq_img_dir
                    mos = load_mos(cfg, eval_ds)
                    images = {r.img_id: r.path for r in __import__("iqa_agent.data", fromlist=["load_images"]).load_images(cfg, eval_ds)}
                    ids = sorted(set(images.keys()) & set(mos.keys()))
                    random.seed(42)
                    sample = sorted(random.sample(ids, min(200, len(ids))))
                    print(f"    抽样: {len(sample)} 张")

                    # 调 API 打分
                    expert_api = {}
                    for sk in POOL:
                        prompt = build_skill_prompt(sk, ds)
                        async def one(img_id, prompt=prompt, sk=sk):
                            try:
                                text, _ = await client.score_image(images[img_id], prompt, temperature=0.0)
                                p = parse_score(text, (lo, hi))
                                return img_id, sk, p["score"] if p else None
                            except Exception as e:
                                return img_id, sk, None

                        jobs = [one(iid) for iid in sample]
                        raw = await gather_with_progress(jobs, every=50, label=f"API-{ds}-{sk[:4]}")
                        for r in raw:
                            if not isinstance(r, Exception) and r[2] is not None:
                                expert_api.setdefault(r[0], {})[r[1]] = r[2]
                        print(f"      {sk}: {sum(1 for r in raw if not isinstance(r,Exception) and r[2] is not None)}/{len(sample)} 张")

                    # API 打分结果 vs 缓存专家分对比
                    cached_expert = load_expert_scores(
                        os.path.join(cfg.runs_dir, "final", r2_dir, "scores.csv"), POOL)
                    diffs = []
                    for iid in sample:
                        if iid in expert_api and iid in cached_expert:
                            for sk in POOL:
                                api_v = expert_api[iid].get(sk)
                                cache_v = cached_expert[iid].get(sk)
                                if api_v is not None and cache_v is not None:
                                    diffs.append(abs(api_v - cache_v))
                    if diffs:
                        print(f"      专家分 |API-缓存|: 均值={np.mean(diffs):.3f}  中位数={np.median(diffs):.3f}  最大={np.max(diffs):.3f}")
                        print(f"      专家分相关性: (见下方管线融合后SRCC对比)")

                    # 用 API 专家分走管线
                    paras = jload(os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json"))
                    rows_api = []
                    for iid in sample:
                        if iid not in expert_api: continue
                        es = expert_api[iid]
                        s3 = [es.get(sk) for sk in POOL]
                        if any(v is None for v in s3): continue
                        sp = float(np.std(s3))
                        fp = images.get(iid)
                        if not fp or not os.path.exists(fp): continue
                        img = Image.open(fp)
                        if ds == "spaq":
                            img.thumbnail((1568, 1568), Image.BICUBIC)
                        f_raw = opencv_features(img)
                        f = np.array([f_raw["lap_var"], f_raw["noise"], f_raw["colorful"],
                                      f_raw["bright"], f_raw["logpix"], f_raw["aspect"], sp], dtype=float)
                        for j, k in enumerate(FEAT_KEYS):
                            if k in ("lap_var", "noise", "colorful"):
                                f[j] = np.log(max(f[j], 1e-6))
                        if ds == "koniq":
                            g = softmax(W @ ((f - mu) / sd))
                            fus = float(g @ np.array(s3))
                        else:
                            fus = float(np.mean(s3))
                        # 用缓存投票分（重新调 4 条 para 成本太高）
                        vote = r6_bare_vote(iid, load_pred(os.path.join(cfg.runs_dir, "final", f"r1b_{ds}", "scores.csv")), paras)
                        if vote is None: continue
                        rows_api.append({"img_id": iid, "score": round(alpha * fus + (1 - alpha) * vote, 4)})

                    if rows_api:
                        m_api = compute_metrics({r["img_id"]: r["score"] for r in rows_api}, mos)
                        print(f"      API 管线结果 ({len(rows_api)}张): SRCC={m_api['SRCC']:.4f}  MAE={m_api['MAE']:.4f}")
                        # 对比同一批图的缓存管线
                        cached_pred = load_pred(os.path.join(cfg.runs_dir, "final", r2_dir.replace("r2_", "r6_"), "scores.csv"))
                        if ds == "spaq":
                            cached_pred = load_pred(os.path.join(cfg.runs_dir, "posthoc", "r6_unified_spaq", "scores.csv"))
                        common = sorted(set(r["img_id"] for r in rows_api) & set(cached_pred.keys()))
                        if common:
                            cp = {iid: cached_pred[iid] for iid in common}
                            pp = {r["img_id"]: r["score"] for r in rows_api if r["img_id"] in common}
                            cm = compute_metrics(cp, mos)
                            pm = compute_metrics(pp, mos)
                            print(f"      同一批图缓存管线: SRCC={cm['SRCC']:.4f}  MAE={cm['MAE']:.4f}")
                            print(f"      API vs 缓存差值:  SRCC {pm['SRCC']-cm['SRCC']:+.4f}  MAE {pm['MAE']-cm['MAE']:+.4f}")

                print(f"\n  账本: {client.ledger()}")

            asyncio.run(api_verify())

    # ═══════════════ HTML 报告 ═══════════════
    if args.html:
        html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verify_report.html")
        html = build_html(table_data, cfg)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  HTML 报告: {os.path.abspath(html_path)}")


def build_html(table_data, cfg):
    rows_html = ""
    for name, sk, mk, ss, ms, src, desc in table_data:
        if sk is None:
            rows_html += f"<tr><td>{name}</td><td colspan='4' class='miss'>数据缺失</td></tr>"
        else:
            rows_html += (
                f"<tr><td>{name}</td>"
                f"<td>{sk:.4f}</td><td>{mk:.4f}</td>"
                f"<td>{ss:.4f}</td><td>{ms:.4f}</td>"
                f"<td class='src'>{src}</td></tr>"
            )

    # Embed figures as base64
    import base64
    figs = {}
    for name in ["fig4_scatter.png", "fig3_w.png"]:
        fp = os.path.join(cfg.runs_dir, "..", "docs", "figs", name)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                figs[name] = base64.b64encode(f.read()).decode()

    fig4_b64 = figs.get("fig4_scatter.png", "")
    fig3_b64 = figs.get("fig3_w.png", "")

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>IQA Agent 复现验证报告</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; background: #fcfcfb; }}
  h1 {{ font-size: 24px; border-bottom: 2px solid #1b2a4a; padding-bottom: 8px; }}
  h2 {{ font-size: 18px; color: #1b2a4a; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #e1e0d9; font-size: 14px; }}
  th {{ background: #1b2a4a; color: #fff; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .miss {{ color: #d03b3b; }}
  .src {{ color: #757575; font-size: 12px; }}
  .note {{ color: #757575; font-size: 13px; margin: 8px 0; }}
  .highlight {{ font-weight: 700; color: #2a78d6; }}
  img {{ max-width: 100%; margin: 12px 0; }}
</style>
</head>
<body>
<h1>IQA Agent 框架 — 复现验证报告</h1>
<p class="note">生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | Backbone: Qwen3-VL-32B | 全部数据来自磁盘缓存 scores.csv</p>

<h2>主表 (8 臂 × 2 数据集)</h2>
<table>
<tr><th>臂</th><th>KonIQ SRCC ↑</th><th>KonIQ MAE ↓</th><th>SPAQ SRCC ↑</th><th>SPAQ MAE ↓</th><th>数据来源</th></tr>
{rows_html}
</table>

<p class="note">R1-bare 至 R6 数据源: runs/final/。R1-anchor v3 与 8B R1-bare 数据源: runs/posthoc/。</p>

<h2>R6 统一框架管线</h2>
<p class="note">KonIQ: W(3×7) 门控矩阵动态融合 (runs/router_v3/fusion_koniq.json) + 裸问释义投票，α=0.6。<br>
SPAQ: 等权回退融合 + 裸问释义投票，α=0.3 (跨域门控未通过一致性检验)。</p>

<h2>预测分 vs 人群分</h2>
{"<img src='data:image/png;base64," + fig4_b64 + "' alt='散点图'>" if fig4_b64 else "<p class='note'>散点图缺失</p>"}

<h2>门控矩阵 W(3×7)</h2>
{"<img src='data:image/png;base64," + fig3_b64 + "' alt='W矩阵'>" if fig3_b64 else "<p class='note'>W矩阵图缺失</p>"}

<p class="note" style="margin-top:40px;">本报告由 <code>python scripts/verify.py --html</code> 自动生成。所有指标来自 scores.csv，未调 API。</p>
</body>
</html>"""


if __name__ == "__main__":
    main()
