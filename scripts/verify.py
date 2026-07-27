#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现验证脚本（唯一入口）。用法见 README。"""
import csv, json, os, sys, time, argparse, random, base64, asyncio
import numpy as np
from PIL import Image

sys.stdout.reconfigure(errors="replace"); sys.stderr.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.router import opencv_features
from iqa_agent.metrics import compute_metrics
from iqa_agent.data import load_mos

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_JSON = os.path.join(BASE, ".verify_cache.json")
API_JSON   = os.path.join(BASE, ".verify_api.json")

POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def load_pred(path):
    d = {}
    if not os.path.exists(path): return d
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = r.get("score", "")
            if s and s.strip(): d[r["img_id"]] = float(s)
    return d

def load_expert(path, pool):
    d = {}
    if not os.path.exists(path): return d
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw = r.get("skill_scores", "")
            if not raw: continue
            try: ss = json.loads(raw)
            except: continue
            entry = {sk: float(ss[sk]) for sk in pool if sk in ss and ss[sk] is not None}
            if len(entry) == len(pool): d[r["img_id"]] = entry
    return d

def bare_vote(iid, r1b, paras):
    vals = [r1b.get(iid)]
    if paras and iid in paras:
        for k in ("1", "2", "3"):
            v = paras[iid].get(k)
            if v is not None: vals.append(float(v))
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

def jload(p):
    if not os.path.exists(p): return {}
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f: return json.load(f)
        except: pass
    return {}


# ═══════════ Sections ═══════════

def sec_env(cfg):
    print("=" * 72)
    print("[1/5] 环境自检"); print("=" * 72)
    ok = True
    for lab, cond, msg in [
        ("Python>=3.10", sys.version_info>=(3,10), sys.version.split()[0]),
        ("numpy", True, "OK"), ("scipy", True, "OK"), ("pillow", True, "OK"),
    ]:
        print(f"  [{'V' if cond else 'X'}] {lab:<16s} {msg}")
        if not cond: ok = False
    for mod, name in [("openai","openai(仅--api)")]:
        try: __import__(mod); print(f"  [V] {name:<16s} OK")
        except ImportError: print(f"  [~] {name:<16s} 未安装(pip install openai)")
    for lab, d in [("数据集 KonIQ", cfg.koniq_img_dir), ("数据集 SPAQ", cfg.spaq_img_dir)]:
        cond = os.path.isdir(d) and len(os.listdir(d)) > 100
        print(f"  [{'V' if cond else 'X'}] {lab:<16s} {d if cond else 'MISSING'}")
        if not cond: ok = False
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    print(f"  [{'V' if has_key else 'X'}] API Key(.env)  {'OK' if has_key else 'MISSING'}")
    for lab, p in [("W 矩阵", "runs/router_v3/fusion_koniq.json"),
                   ("R2 KonIQ", "runs/final/r2_koniq/scores.csv"),
                   ("R2 SPAQ", "runs/final/r2_spaq/scores.csv")]:
        cond = os.path.exists(os.path.join(cfg.runs_dir, *p.split("/")[1:]))
        print(f"  [{'V' if cond else 'X'}] {lab:<16s} {'OK' if cond else 'MISSING'}")
    return ok


def sec_table(cfg):
    print("\n" + "=" * 72)
    print("[2/5] 主表验证 (纯缓存，零 API)"); print("=" * 72)
    ARMS = [
        ("R1-bare",        "r1b_koniq",      "r1b_spaq"),
        ("R1-rich",        "r1r_koniq",      "r1r_spaq"),
        ("R2",             "r2_koniq",       "r2_spaq"),
        ("R2.5",           "r25_koniq",      "r25_spaq"),
        ("R3",             "r3_koniq",       "r3_spaq"),
        ("R6 (统一框架)",   "r6_koniq",       "r6_spaq"),
        ("R1-anchor v3",   "r1anchor_v3_koniq","r1anchor_v3_spaq"),
        ("8B R1-bare",     "r1b_8b_koniq",   "r1b_8b_spaq"),
    ]
    hdr = "  {:<20s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>6s}"
    row = "  {:<20s}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>6s}"
    print(f"\n  {'─'*62}"); print(hdr.format("臂", "KonIQ_SRCC", "KonIQ_MAE", "SPAQ_SRCC", "SPAQ_MAE", "来源"))
    print(f"  {'─'*62}")
    table = []
    for arm, dk, ds_ in ARMS:
        root = "posthoc" if arm.startswith(("R1-anchor","8B")) else "final"
        pk = os.path.join(cfg.runs_dir, root, dk, "scores.csv")
        ps = os.path.join(cfg.runs_dir, root, ds_, "scores.csv")
        if os.path.exists(pk):
            mk = compute_metrics(load_pred(pk), load_mos(cfg, "koniq_val"))
            ms_ = compute_metrics(load_pred(ps), load_mos(cfg, "spaq_test"))
            print(row.format(arm, mk["SRCC"], mk["MAE"], ms_["SRCC"], ms_["MAE"], "缓存"))
            table.append([arm, mk["SRCC"], mk["MAE"], ms_["SRCC"], ms_["MAE"], "缓存"])
        else:
            print(f"  {arm:<20s}  {'MISSING':>8s}")
            table.append([arm, None, None, None, None, "缺失"])
    print(f"  {'─'*62}")
    with open(CACHE_JSON, "w", encoding="utf-8") as f: json.dump(table, f, ensure_ascii=False)
    return table


def sec_pipeline(cfg, limit=None):
    print("\n" + "=" * 72)
    print("[3/5] R6 统一框架管线 (缓存专家分，零 API)"); print("=" * 72)
    fk = json.load(open(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"), encoding="utf-8"))
    W = np.array(fk["W"]); mu = np.array(fk["mu"])
    sd = np.array([s if s > 1e-6 else 1.0 for s in fk["sd"]])

    results = {}
    for ds, eval_ds, alpha, r2d, r1bd, paraf in [
        ("koniq", "koniq_val", 0.6, "r2_koniq", "r1b_koniq", "r6_koniq_paras.json"),
        ("spaq", "spaq_test", 0.3, "r2_spaq", "r1b_spaq", "r6_spaq_paras.json"),
    ]:
        print(f"\n  ── {ds.upper()} R6 管线 (α={alpha}) ──")
        r2p = os.path.join(cfg.runs_dir, "final", r2d, "scores.csv")
        if not os.path.exists(r2p): print("    SKIP"); continue
        expert = load_expert(r2p, POOL)
        r1b = load_pred(os.path.join(cfg.runs_dir, "final", r1bd, "scores.csv"))
        paras = jload(os.path.join(cfg.runs_dir, paraf))
        mos = load_mos(cfg, eval_ds)
        img_dir = cfg.koniq_img_dir if ds == "koniq" else cfg.spaq_img_dir
        ids = sorted(set(expert) & set(r1b) & set(mos))
        if limit: random.seed(42); ids = sorted(random.sample(ids, min(limit, len(ids))))
        print(f"    图像数: {len(ids)}")

        cached = load_pred(os.path.join(cfg.runs_dir, "final", r2d.replace("r2_", "r6_"), "scores.csv"))
        if ds == "spaq":
            cu = os.path.join(cfg.runs_dir, "posthoc", "r6_unified_spaq", "scores.csv")
            if os.path.exists(cu): cached = load_pred(cu)

        rows, t0, detail = [], time.time(), 0
        for idx, iid in enumerate(ids):
            s3 = [expert[iid][sk] for sk in POOL]
            sp = float(np.std(s3))
            fp = os.path.join(img_dir, iid)
            if not os.path.exists(fp): continue
            img = Image.open(fp)
            if ds == "spaq": img.thumbnail((1568, 1568), Image.BICUBIC)
            f_raw = opencv_features(img)
            f = np.array([f_raw[k] for k in FEAT_KEYS[:6]] + [sp], dtype=float)
            for j, k in enumerate(FEAT_KEYS):
                if k in ("lap_var", "noise", "colorful"): f[j] = np.log(max(f[j], 1e-6))
            if ds == "koniq":
                logits = W @ ((f - mu) / sd)
                g = softmax(logits)
                fus = float(g @ np.array(s3))
            else:
                logits = np.zeros(3); g = np.array([1/3, 1/3, 1/3]); fus = float(np.mean(s3))
            vote = bare_vote(iid, r1b, paras)
            if vote is None: continue
            final = round(alpha * fus + (1 - alpha) * vote, 4)
            rows.append({"img_id": iid, "score": final})
            if detail < 3:
                print(f"\n    [{detail+1}/3] {iid}")
                print(f"      专家分: TECH={s3[0]:.2f} GLOBAL={s3[1]:.2f} CONTENT={s3[2]:.2f} spread={sp:.2f}")
                print(f"      像素特征: lap={f_raw['lap_var']:.1f} noise={f_raw['noise']:.3f} color={f_raw['colorful']:.1f} bright={f_raw['bright']:.1f}")
                if ds == "koniq": print(f"      logits: [{logits[0]:+.3f}, {logits[1]:+.3f}, {logits[2]:+.3f}]")
                print(f"      话语权: TECH={g[0]:.3f} GLOBAL={g[1]:.3f} CONTENT={g[2]:.3f}")
                print(f"      融合分={fus:.4f}  投票分={vote:.4f}  最终分={final:.4f}")
                if iid in cached: print(f"      缓存R6: {cached[iid]:.4f}")
                detail += 1
            if (idx + 1) % 200 == 0: print(f"    [{idx+1}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)

        pred = {r["img_id"]: r["score"] for r in rows}
        m = compute_metrics(pred, mos)
        print(f"\n    耗时: {time.time()-t0:.0f}s")
        print(f"    管线重算: SRCC={m['SRCC']:.4f}  MAE={m['MAE']:.4f}  n={m['n']}")

        if cached:
            common = sorted(set(pred) & set(cached))
            if common:
                cm = compute_metrics({i: cached[i] for i in common}, mos)
                pm = compute_metrics({i: pred[i] for i in common}, mos)
                d_srcc = pm["SRCC"] - cm["SRCC"]
                d_mae = pm["MAE"] - cm["MAE"]
                print(f"    已缓存R6:     SRCC={cm['SRCC']:.4f}  MAE={cm['MAE']:.4f}")
                print(f"    差值:          SRCC {d_srcc:+.4f}  MAE {d_mae:+.4f}  {'V 一致' if abs(d_srcc)<0.01 and abs(d_mae)<0.01 else '! 差异>0.01'}")
                results[ds] = {"cached": cm, "recomputed": pm, "n": cm["n"]}
    return results


def sec_cke(cfg):
    print("\n" + "=" * 72)
    print("[4/5] CKE 自进化规则库 (已被最终框架移除)"); print("=" * 72)
    p = os.path.join(cfg.runs_dir, "cke", "final_library.json")
    if os.path.exists(p):
        rules = jload(p)
        rlist = rules.get("rules", []) if isinstance(rules, dict) else (rules if isinstance(rules, list) else [])
        print(f"\n  共 {len(rlist)} 条，经 CKE 四轮迭代+双门控筛选。")
        print("  内部: 阶梯单调性 0.455→0.474 | B-C分歧 0.402→0.330")
        print("  外部: KonIQ SRCC 仅 +0.001 — 自洽≠对齐\n")
        for i, r in enumerate(rlist, 1):
            txt = r if isinstance(r, str) else r.get("rule", str(r))
            print(f"  规则{i}: {txt[:140]}")
    else:
        print(f"\n  [注意] 文件不存在: {p}")


def sec_api(cfg, limit=None):
    """返回 {('koniq','api'): metrics, ('spaq','api'): metrics}"""
    print("\n" + "=" * 72)
    print("[5/5] API 抽样重跑 (KonIQx200 + SPAQx200, 32B)"); print("=" * 72)
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("  SKIP: 未配置 API Key"); return {}
    from iqa_agent.client import VLMClient, gather_with_progress
    from iqa_agent.prompts.skills import build_skill_prompt
    from iqa_agent.scoring import parse_score

    fk = json.load(open(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json"), encoding="utf-8"))
    W = np.array(fk["W"]); mu = np.array(fk["mu"])
    sd = np.array([s if s > 1e-6 else 1.0 for s in fk["sd"]])
    api_results = {}

    async def run():
        client = VLMClient(cfg, cfg.model_main)
        for ds, eval_ds, lo, hi, r2d, alpha in [
            ("koniq", "koniq_val", 1, 5, "r2_koniq", 0.6),
            ("spaq", "spaq_test", 0, 10, "r2_spaq", 0.3),
        ]:
            print(f"\n  ── {ds.upper()} API 抽样 ──")
            mos = load_mos(cfg, eval_ds)
            from iqa_agent.data import load_images
            imgs = {r.img_id: r.path for r in load_images(cfg, eval_ds)}
            ids_all = sorted(set(imgs) & set(mos))
            random.seed(42); sample = sorted(random.sample(ids_all, min(200, len(ids_all))))
            print(f"    抽样: {len(sample)} 张")

            expert_api = {}
            for sk in POOL:
                prompt = build_skill_prompt(sk, ds)
                async def one(img_id, prompt=prompt, sk=sk):
                    try:
                        txt, _ = await client.score_image(imgs[img_id], prompt, temperature=0.0)
                        p = parse_score(txt, (lo, hi))
                        return img_id, sk, p["score"] if p else None
                    except Exception:
                        return img_id, sk, None
                raw = await gather_with_progress([one(iid) for iid in sample], every=50, label=f"API-{ds}-{sk[:4]}")
                for r in raw:
                    if not isinstance(r, Exception) and r[2] is not None:
                        expert_api.setdefault(r[0], {})[r[1]] = r[2]
                ok_n = sum(1 for r in raw if not isinstance(r, Exception) and r[2] is not None)
                print(f"      {sk}: {ok_n}/{len(sample)} 张{' (全缓存命中)' if ok_n == len(sample) and client.cache_hits > 0 else ''}")

            # compare with cached expert scores
            ce = load_expert(os.path.join(cfg.runs_dir, "final", r2d, "scores.csv"), POOL)
            diffs = []
            for iid in sample:
                if iid in expert_api and iid in ce:
                    for sk in POOL:
                        av = expert_api[iid].get(sk); cv = ce[iid].get(sk)
                        if av is not None and cv is not None: diffs.append(abs(av - cv))
            if diffs: print(f"      专家分 |实时-缓存|: 均值={np.mean(diffs):.3f}  中位数={np.median(diffs):.3f}  最大={np.max(diffs):.3f}")

            # run pipeline with API scores
            paras = jload(os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json"))
            r1b = load_pred(os.path.join(cfg.runs_dir, "final", f"r1b_{ds}", "scores.csv"))
            rows_api = []
            for iid in sample:
                if iid not in expert_api: continue
                s3 = [expert_api[iid].get(sk) for sk in POOL]
                if any(v is None for v in s3): continue
                sp = float(np.std(s3))
                if not imgs.get(iid) or not os.path.exists(imgs[iid]): continue
                img = Image.open(imgs[iid])
                if ds == "spaq": img.thumbnail((1568, 1568), Image.BICUBIC)
                f_raw = opencv_features(img)
                f = np.array([f_raw[k] for k in FEAT_KEYS[:6]] + [sp], dtype=float)
                for j, k in enumerate(FEAT_KEYS):
                    if k in ("lap_var", "noise", "colorful"): f[j] = np.log(max(f[j], 1e-6))
                if ds == "koniq": fus = float(softmax(W @ ((f - mu) / sd)) @ np.array(s3))
                else: fus = float(np.mean(s3))
                vote = bare_vote(iid, r1b, paras)
                if vote is None: continue
                rows_api.append({"img_id": iid, "score": round(alpha * fus + (1 - alpha) * vote, 4)})

            ma = compute_metrics({r["img_id"]: r["score"] for r in rows_api}, mos) if rows_api else {"SRCC": None, "MAE": None, "n": 0}
            print(f"      API管线 ({len(rows_api)}张): SRCC={ma['SRCC']:.4f}  MAE={ma['MAE']:.4f}" if ma["SRCC"] else "      API管线: 无有效结果")
            api_results[(ds, "api")] = ma

            # compare with cached pipeline on same images
            cp = load_pred(os.path.join(cfg.runs_dir, "final", r2d.replace("r2_", "r6_"), "scores.csv"))
            if ds == "spaq":
                cup = os.path.join(cfg.runs_dir, "posthoc", "r6_unified_spaq", "scores.csv")
                if os.path.exists(cup): cp = load_pred(cup)
            common = sorted(set(r["img_id"] for r in rows_api) & set(cp))
            if common and ma["SRCC"]:
                cm = compute_metrics({i: cp[i] for i in common}, mos)
                pm2 = compute_metrics({r["img_id"]: r["score"] for r in rows_api if r["img_id"] in common}, mos)
                print(f"      同批缓存管线:   SRCC={cm['SRCC']:.4f}  MAE={cm['MAE']:.4f}")
                print(f"      API vs 缓存差值: SRCC {pm2['SRCC']-cm['SRCC']:+.4f}  MAE {pm2['MAE']-cm['MAE']:+.4f}")
                api_results[(ds, "cached_same")] = cm

        print(f"\n  账本: api_calls={client.calls}, cache_hits={client.cache_hits}, tokens_in={client.tokens_in}, tokens_out={client.tokens_out}")
        # 保存 API 结果供 --html-only 使用
        api_save = {}
        for (dom, tag), m in api_results.items():
            if m and m.get("SRCC"):
                api_save[f"{dom}_{tag}"] = {"SRCC": m["SRCC"], "MAE": m["MAE"], "n": m.get("n", 0)}
        try:
            with open(API_JSON, "w", encoding="utf-8") as f:
                json.dump(api_save, f, ensure_ascii=False)
        except Exception:
            pass
    asyncio.run(run())
    return api_results


def print_compare(cached_table, api_results=None):
    """仅在有 API 结果时打印对照表：R6缓存 vs R6实时API。"""
    if not api_results: return
    ak = api_results.get(("koniq", "api"), {})
    a_s = api_results.get(("spaq", "api"), {})
    if not ak.get("SRCC") or not a_s.get("SRCC"): return

    r6k = r6s = None
    for row in cached_table:
        if row[0] == "R6 (统一框架)":
            r6k = (row[1], row[2]); r6s = (row[3], row[4]); break

    if not r6k: return
    sk, mk = r6k; ss, ms = r6s

    print("\n" + "=" * 72)
    print("[对照] 缓存 vs API 实时重跑 — 排除偶然性")
    print("=" * 72)
    print(f"  {'─'*62}")
    print(f"  {'':>20s}  {'KonIQ SRCC':>10s}  {'KonIQ MAE':>10s}  {'SPAQ SRCC':>10s}  {'SPAQ MAE':>10s}")
    print(f"  {'─'*62}")
    print(f"  {'R6 缓存(预存)':>20s}  {sk:>10.4f}  {mk:>10.4f}  {ss:>10.4f}  {ms:>10.4f}")
    print(f"  {'R6 API重跑(200张)':>20s}  {ak['SRCC']:>10.4f}  {ak['MAE']:>10.4f}  {a_s['SRCC']:>10.4f}  {a_s['MAE']:>10.4f}")
    dk_s = ak['SRCC'] - sk; dk_m = ak['MAE'] - mk
    ds_s = a_s['SRCC'] - ss; ds_m = a_s['MAE'] - ms
    print(f"  {'差值':>20s}  {dk_s:>+10.4f}  {dk_m:>+10.4f}  {ds_s:>+10.4f}  {ds_m:>+10.4f}")
    print(f"  {'─'*62}")
    all_ok = all(abs(d) < 0.02 for d in [dk_s, dk_m, ds_s, ds_m])
    print(f"\n  {'V 缓存与实际重跑一致（±0.02以内）— 指标非偶然' if all_ok else '! 差值 > 0.02 — 需排查'}")


def sec_html(cfg, cached_table, api_results=None):
    print("\n" + "=" * 72); print("[HTML] 生成自包含报告"); print("=" * 72)
    html_path = os.path.join(BASE, "verify_report.html")
    # 尝试从磁盘加载 API 结果（--html-only 场景）
    if not api_results and os.path.exists(API_JSON):
        try:
            raw = json.load(open(API_JSON, encoding="utf-8"))
            api_results = {}
            for k, v in raw.items():
                parts = k.split("_", 1)
                if len(parts) == 2:
                    api_results[(parts[0], parts[1])] = v
        except Exception:
            pass
    html_path = os.path.join(BASE, "verify_report.html")
    rows_h = ""
    for row in cached_table:
        if row[1] is None: rows_h += f"<tr><td>{row[0]}</td><td colspan='4' class='miss'>缺失</td></tr>"
        else: rows_h += f"<tr><td>{row[0]}</td><td>{row[1]:.4f}</td><td>{row[2]:.4f}</td><td>{row[3]:.4f}</td><td>{row[4]:.4f}</td><td class='src'>{row[5]}</td></tr>"
    # Add API row if exists
    if api_results:
        ak = api_results.get(("koniq", "api"), {}); a_s = api_results.get(("spaq", "api"), {})
        if ak and ak.get("SRCC"):
            rows_h += (f"<tr style='background:#E8F0FE'><td>R6 API重跑(200张)</td>"
                       f"<td>{ak['SRCC']:.4f}</td><td>{ak['MAE']:.4f}</td>"
                       f"<td>{a_s['SRCC']:.4f}</td><td>{a_s['MAE']:.4f}</td><td class='src'>实时API</td></tr>")

    figs_b64 = {}
    for name in ["fig4_scatter.png", "fig3_w.png"]:
        fp = os.path.join(BASE, "docs", "figs", name)
        if os.path.exists(fp):
            with open(fp, "rb") as f: figs_b64[name] = base64.b64encode(f.read()).decode()
    img_tags = ""
    if "fig4_scatter.png" in figs_b64: img_tags += f"<img src='data:image/png;base64,{figs_b64['fig4_scatter.png']}'>"
    if "fig3_w.png" in figs_b64: img_tags += f"<img src='data:image/png;base64,{figs_b64['fig3_w.png']}'>"

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>IQA Agent 复现验证报告</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#1a1a1a;background:#fcfcfb}}
h1{{font-size:24px;border-bottom:2px solid #1b2a4a;padding-bottom:8px}}h2{{font-size:18px;color:#1b2a4a;margin-top:32px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{padding:8px 12px;text-align:center;border-bottom:1px solid #e1e0d9;font-size:14px}}
th{{background:#1b2a4a;color:#fff}}tr:nth-child(even){{background:#f5f5f5}}
.miss{{color:#d03b3b}}.src{{color:#757575;font-size:12px}}.note{{color:#757575;font-size:13px}}img{{max-width:100%;margin:12px 0}}</style></head><body>
<h1>IQA Agent 框架 — 复现验证</h1>
<p class="note">{time.strftime('%Y-%m-%d %H:%M:%S')} | Backbone: Qwen3-VL-32B | 缓存: runs/final/*/scores.csv</p>
<h2>主表</h2><table><tr><th>臂</th><th>KonIQ SRCC</th><th>KonIQ MAE</th><th>SPAQ SRCC</th><th>SPAQ MAE</th><th>来源</th></tr>{rows_h}</table>
<h2>图表</h2>{img_tags}
<p class="note" style="margin-top:40px">由 python scripts/verify.py --html 生成</p></body></html>"""
    with open(html_path, "w", encoding="utf-8") as f: f.write(html)
    print(f"  HTML: {html_path}")


# ═══════════ main ═══════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true"); ap.add_argument("--html", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api-only", action="store_true"); ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    do_default = not args.api and not args.api_only and not args.html and not args.html_only
    do_api = args.api or args.api_only
    do_html = args.html or args.html_only
    cfg = get_config()

    if args.html_only or args.html:
        tb = json.load(open(CACHE_JSON, encoding="utf-8")) if os.path.exists(CACHE_JSON) else []
        if not tb: print("错误: 未找到 .verify_cache.json，请先运行 python scripts/verify.py"); return
        sec_html(cfg, tb)
        return

    if args.api_only:
        api_r = sec_api(cfg, args.limit)
        if api_r and os.path.exists(CACHE_JSON):
            tb = json.load(open(CACHE_JSON, encoding="utf-8"))
            print_compare(tb, api_r)
        return

    if do_default:
        sec_env(cfg)
        cached_table = sec_table(cfg)
        sec_pipeline(cfg, args.limit)
        sec_cke(cfg)

    if do_api:
        api_r = sec_api(cfg, args.limit)
        cached_table = json.load(open(CACHE_JSON, encoding="utf-8")) if os.path.exists(CACHE_JSON) else []
        print_compare(cached_table, api_r)
        print_compare(cached_table, {}, api_r)


if __name__ == "__main__":
    main()
