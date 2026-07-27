# -*- coding: utf-8 -*-
r"""
复现验证脚本（唯一入口）。从 GitHub clone 后只需配好 .env 和数据集即可运行。

用法:
  python scripts/verify.py                 零 API，缓存主表 + R6 管线
  python scripts/verify.py --api           同上 + KonIQx200 + SPAQx200 API 重调专家分
  python scripts/verify.py --html          同上 + 输出 HTML 报告
  python scripts/verify.py --api-only      仅跑 API 抽样 (跳过 1-4)
  python scripts/verify.py --html-only     仅生成 HTML 报告 (需先跑过一次默认)
"""
import csv, json, os, sys, time, argparse, random
import numpy as np
from PIL import Image

sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iqa_agent.config import get_config
from iqa_agent.router import opencv_features
from iqa_agent.metrics import compute_metrics
from iqa_agent.data import load_mos

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP  = os.path.join(BASE, ".verify_table.json")


# ═══════════════ 工具函数 ═══════════════
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def jload(p):
    if not os.path.exists(p): return {}
    for enc in ("utf-8","gbk"):
        try:
            with open(p,encoding=enc) as f: return json.load(f)
        except: pass
    return {}

def load_pred(path):
    pred = {}
    if not os.path.exists(path): return pred
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = r.get("score","")
            if s and s.strip(): pred[r["img_id"]] = float(s)
    return pred

def load_expert(r2_path, pool):
    expert = {}
    if not os.path.exists(r2_path): return expert
    with open(r2_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ss_raw = r.get("skill_scores","")
            if not ss_raw: continue
            try: ss = json.loads(ss_raw)
            except: continue
            entry = {sk: float(ss[sk]) for sk in pool if sk in ss and ss[sk] is not None}
            if len(entry) == len(pool): expert[r["img_id"]] = entry
    return expert

def vote_mean(iid, r1b, paras):
    vals = [r1b.get(iid)]
    if paras and iid in paras:
        for k in ("1","2","3"):
            v = paras[iid].get(k)
            if v is not None: vals.append(float(v))
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# ═══════════════ Section 实现 ═══════════════

def sec1_selfcheck(cfg):
    print("=" * 72)
    print("[1/5] 环境自检")
    print("=" * 72)
    checks = []
    checks.append(("Python >= 3.10", sys.version_info >= (3,10), sys.version.split()[0]))
    for mod, name in [("numpy","numpy"), ("scipy.stats","scipy"), ("PIL","pillow")]:
        try: __import__(mod); checks.append((name,True,"OK"))
        except ImportError: checks.append((name,False,"MISSING"))
    try: __import__("openai"); checks.append(("openai",True,"OK"))
    except ImportError: checks.append(("openai",False,"MISSING (仅--api需要)"))
    for lab, d in [("KonIQ",cfg.koniq_img_dir), ("SPAQ",cfg.spaq_img_dir)]:
        ok = os.path.isdir(d) and len(os.listdir(d)) > 100
        checks.append((f"数据集 {lab}", ok, d if ok else f"MISSING — {d}"))
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    checks.append(("API Key", has_key, "OK" if has_key else "MISSING"))
    for lab, p in [("W 矩阵","runs/router_v3/fusion_koniq.json"),
                   ("R2 KonIQ 专家","runs/final/r2_koniq/scores.csv"),
                   ("R2 SPAQ 专家","runs/final/r2_spaq/scores.csv")]:
        ok = os.path.exists(os.path.join(cfg.runs_dir, *p.split("/")[1:]))
        checks.append((lab, ok, "OK" if ok else "MISSING"))
    for name, ok, detail in checks:
        print(f"  [{'V' if ok else 'X'}] {name:<20s} {detail}")
    return all(ok for _, ok, _ in checks)


def sec2_table(cfg):
    print("\n" + "=" * 72)
    print("[2/5] 主表验证 (纯缓存，零 API)")
    print("=" * 72)
    ARMS = [
        ("R1-bare",        "r1b_koniq",         "r1b_spaq",         ""),
        ("R1-rich",        "r1r_koniq",         "r1r_spaq",         ""),
        ("R2",             "r2_koniq",          "r2_spaq",          ""),
        ("R2.5",           "r25_koniq",         "r25_spaq",         ""),
        ("R3",             "r3_koniq",          "r3_spaq",          ""),
        ("R6 (统一框架)",   "r6_koniq",          "r6_spaq",          ""),
        ("R1-anchor v3",   "r1anchor_v3_koniq", "r1anchor_v3_spaq",""),
        ("8B R1-bare",     "r1b_8b_koniq",      "r1b_8b_spaq",     ""),
    ]
    hdr = "  {:<20s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>6s}"
    row_fmt = "  {:<20s}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>6s}"
    print(f"\n  {'─'*62}")
    print(hdr.format("臂","KonIQ_SRCC","KonIQ_MAE","SPAQ_SRCC","SPAQ_MAE","来源"))
    print(f"  {'─'*62}")
    table = []
    for arm, dk, ds_, _desc in ARMS:
        root_k = cfg.runs_dir if arm.startswith(("R"," 8")) else os.path.join(cfg.runs_dir,"posthoc")
        pk = os.path.join(cfg.runs_dir if not arm.startswith(("R1-anchor","8B")) else os.path.join(cfg.runs_dir,"posthoc"), dk, "scores.csv")
        ps_ = os.path.join(cfg.runs_dir if not arm.startswith(("R1-anchor","8B")) else os.path.join(cfg.runs_dir,"posthoc"), ds_, "scores.csv")
        pk = os.path.join(cfg.runs_dir, "final" if not arm.startswith(("R1-anchor","8B")) else "posthoc", dk, "scores.csv")
        ps_ = os.path.join(cfg.runs_dir, "final" if not arm.startswith(("R1-anchor","8B")) else "posthoc", ds_, "scores.csv")
        if os.path.exists(pk):
            m_k = compute_metrics(load_pred(pk), load_mos(cfg,"koniq_val"))
            m_s = compute_metrics(load_pred(ps_), load_mos(cfg,"spaq_test"))
            print(row_fmt.format(arm, m_k["SRCC"], m_k["MAE"], m_s["SRCC"], m_s["MAE"], "缓存"))
            table.append([arm, m_k["SRCC"], m_k["MAE"], m_s["SRCC"], m_s["MAE"], "缓存"])
        else:
            print(f"  {arm:<20s}  {'MISSING':>8s}")
            table.append([arm, None, None, None, None, "缺失"])
    print(f"  {'─'*62}")
    with open(TMP,"w",encoding="utf-8") as f: json.dump(table,f,ensure_ascii=False)
    return table


def sec3_pipeline(cfg, limit=None):
    print("\n" + "=" * 72)
    print("[3/5] R6 统一框架管线 (缓存专家分，零 API)")
    print("=" * 72)
    POOL = ["S-TECH","S-GLOBAL","S-CONTENT"]
    FEAT_KEYS = ["lap_var","noise","colorful","bright","logpix","aspect","spread"]
    fk = json.load(open(os.path.join(cfg.runs_dir,"router_v3","fusion_koniq.json"), encoding="utf-8"))
    W = np.array(fk["W"]); mu = np.array(fk["mu"])
    sd = np.array([s if s>1e-6 else 1.0 for s in fk["sd"]])

    for ds, eval_ds, alpha, r2_dir, r1b_dir, para_file in [
        ("koniq","koniq_val",0.6,"r2_koniq","r1b_koniq","r6_koniq_paras.json"),
        ("spaq","spaq_test",0.3,"r2_spaq","r1b_spaq","r6_spaq_paras.json")]:
        print(f"\n  ── {ds.upper()} R6 管线 (α={alpha}) ──")
        r2_path = os.path.join(cfg.runs_dir,"final",r2_dir,"scores.csv")
        r1b_path = os.path.join(cfg.runs_dir,"final",r1b_dir,"scores.csv")
        if not os.path.exists(r2_path): print(f"    SKIP: {r2_path} 不存在"); continue
        expert = load_expert(r2_path, POOL)
        r1b = load_pred(r1b_path)
        paras = jload(os.path.join(cfg.runs_dir,para_file))
        mos = load_mos(cfg, eval_ds)
        img_dir = cfg.koniq_img_dir if ds=="koniq" else cfg.spaq_img_dir
        ids_all = sorted(set(expert)&set(r1b)&set(mos))
        if limit: random.seed(42); ids_all = sorted(random.sample(ids_all, min(limit,len(ids_all))))
        print(f"    图像数: {len(ids_all)}")
        cached_path = os.path.join(cfg.runs_dir,"final",r2_dir.replace("r2_","r6_"),"scores.csv")
        cached = load_pred(cached_path)
        if ds == "spaq":
            cu = os.path.join(cfg.runs_dir,"posthoc","r6_unified_spaq","scores.csv")
            if os.path.exists(cu): cached = load_pred(cu)
        rows, t0 = [], time.time()
        detail = 0
        for idx, iid in enumerate(ids_all):
            es = expert[iid]; s3 = [es.get(sk) for sk in POOL]
            if any(v is None for v in s3): continue
            sp = float(np.std(s3))
            fp = os.path.join(img_dir, iid)
            if not os.path.exists(fp): continue
            img = Image.open(fp)
            if ds=="spaq": img.thumbnail((1568,1568), Image.BICUBIC)
            f_raw = opencv_features(img)
            f = np.array([f_raw[k] for k in FEAT_KEYS[:6]]+[sp], dtype=float)
            for j,k in enumerate(FEAT_KEYS):
                if k in ("lap_var","noise","colorful"): f[j]=np.log(max(f[j],1e-6))
            if ds=="koniq":
                logits = W @ ((f-mu)/sd)
                g = softmax(logits)
                fus = float(g @ np.array(s3))
            else:
                logits = np.zeros(3)
                g = np.array([1/3,1/3,1/3])
                fus = float(np.mean(s3))
            vote = vote_mean(iid, r1b, paras)
            if vote is None: continue
            final = round(alpha*fus + (1-alpha)*vote, 4)
            rows.append({"img_id":iid,"score":final})
            if detail < 3:
                print(f"\n    [{detail+1}/3] {iid}")
                print(f"      专家分(缓存): TECH={s3[0]:.2f} GLOBAL={s3[1]:.2f} CONTENT={s3[2]:.2f} spread={sp:.2f}")
                print(f"      像素特征:      lap_var={f_raw['lap_var']:.1f} noise={f_raw['noise']:.3f} colorful={f_raw['colorful']:.1f}")
                print(f"                      bright={f_raw['bright']:.1f} logpix={f_raw['logpix']:.2f} aspect={f_raw['aspect']:.3f}")
                if ds=="koniq": print(f"      门控logits:    [{logits[0]:+.3f}, {logits[1]:+.3f}, {logits[2]:+.3f}]")
                print(f"      话语权 g:      TECH={g[0]:.3f} GLOBAL={g[1]:.3f} CONTENT={g[2]:.3f}")
                print(f"      融合分:        {fus:.4f}  |  投票分: {vote:.4f}")
                print(f"      最终分:        {alpha}*{fus:.4f}+{1-alpha}*{vote:.4f}={final:.4f}")
                if iid in cached: print(f"      已缓存 R6 分:  {cached[iid]:.4f}")
                detail += 1
            if (idx+1)%200==0: print(f"    [{idx+1}/{len(ids_all)}] {time.time()-t0:.0f}s", flush=True)
        pred = {r["img_id"]:r["score"] for r in rows}
        m = compute_metrics(pred, mos)
        print(f"\n    耗时: {time.time()-t0:.0f}s")
        print(f"    管线重算: SRCC={m['SRCC']:.4f} MAE={m['MAE']:.4f} n={m['n']}")
        if cached:
            common = sorted(set(pred)&set(cached))
            if common:
                cm = compute_metrics({i:cached[i] for i in common}, mos)
                pm = compute_metrics({i:pred[i] for i in common}, mos)
                print(f"    已缓存 R6: SRCC={cm['SRCC']:.4f} MAE={cm['MAE']:.4f}")
                print(f"    差值:      SRCC {pm['SRCC']-cm['SRCC']:+.4f}  MAE {pm['MAE']-cm['MAE']:+.4f}")
                if abs(pm['SRCC']-cm['SRCC'])<0.01 and abs(pm['MAE']-cm['MAE'])<0.01:
                    print(f"    V 管线重算与缓存一致")
                else: print(f"    ! 差异 > 0.01")


def sec4_cke(cfg):
    print("\n"+"="*72)
    print("[4/5] CKE 自进化规则库 (已被最终框架移除)")
    print("="*72)
    p = os.path.join(cfg.runs_dir,"cke","final_library.json")
    if os.path.exists(p):
        rules = jload(p)
        if isinstance(rules, dict):
            rlist = rules.get("rules", [])
        elif isinstance(rules, list):
            rlist = rules
        else:
            rlist = []
        print(f"\n  共 {len(rlist)} 条规则，经 CKE 四轮迭代+双门控筛选。")
        print("  内部指标: 阶梯单调性 0.455→0.474 / B-C分歧 0.402→0.330")
        print("  外部结果: KonIQ SRCC 仅 +0.001 — 自洽≠对齐。\n")
        for i, r in enumerate(rlist, 1):
            txt = r if isinstance(r, str) else r.get("rule", str(r))
            print(f"  规则 {i}: {txt[:150]}")
    else:
        print(f"\n  [注意] 规则库文件不存在: {p}")


def sec5_api(cfg, limit=None):
    print("\n" + "=" * 72)
    print("[5/5] API 抽样重跑 (KonIQx200 + SPAQx200，32B)")
    print("=" * 72)
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    if not has_key:
        print("  SKIP: 未配置 API Key")
        return
    from iqa_agent.client import VLMClient, gather_with_progress
    from iqa_agent.prompts.skills import build_skill_prompt
    from iqa_agent.scoring import parse_score
    import asyncio

    POOL = ["S-TECH","S-GLOBAL","S-CONTENT"]
    FEAT_KEYS = ["lap_var","noise","colorful","bright","logpix","aspect","spread"]
    fk = json.load(open(os.path.join(cfg.runs_dir,"router_v3","fusion_koniq.json"), encoding="utf-8"))
    W = np.array(fk["W"]); mu = np.array(fk["mu"])
    sd = np.array([s if s>1e-6 else 1.0 for s in fk["sd"]])

    async def run():
        client = VLMClient(cfg, cfg.model_main)
        for ds, eval_ds, lo, hi, r2_dir, alpha in [
            ("koniq","koniq_val",1,5,"r2_koniq",0.6),
            ("spaq","spaq_test",0,10,"r2_spaq",0.3)]:
            print(f"\n  ── {ds.upper()} API 抽样 ──")
            mos = load_mos(cfg, eval_ds)
            from iqa_agent.data import load_images
            imgs = {r.img_id: r.path for r in load_images(cfg, eval_ds)}
            ids_all = sorted(set(imgs)&set(mos))
            random.seed(42)
            sample = sorted(random.sample(ids_all, min(200, len(ids_all))))
            print(f"    抽样: {len(sample)} 张")
            expert_api = {}
            for sk in POOL:
                prompt = build_skill_prompt(sk, ds)
                async def one(img_id, prompt=prompt, sk=sk):
                    try:
                        txt, _ = await client.score_image(imgs[img_id], prompt, temperature=0.0)
                        p = parse_score(txt, (lo, hi))
                        return img_id, sk, p["score"] if p else None
                    except Exception as e:
                        return img_id, sk, None
                raw = await gather_with_progress([one(iid) for iid in sample], every=50, label=f"API-{ds}-{sk[:4]}")
                for r_ in raw:
                    if not isinstance(r_, Exception) and r_[2] is not None:
                        expert_api.setdefault(r_[0], {})[r_[1]] = r_[2]
                ok_n = sum(1 for r_ in raw if not isinstance(r_,Exception) and r_[2] is not None)
                print(f"      {sk}: {ok_n}/{len(sample)} 张")
            ce = load_expert(os.path.join(cfg.runs_dir,"final",r2_dir,"scores.csv"), POOL)
            diffs = []
            for iid in sample:
                if iid in expert_api and iid in ce:
                    for sk in POOL:
                        av = expert_api[iid].get(sk); cv = ce[iid].get(sk)
                        if av is not None and cv is not None: diffs.append(abs(av-cv))
            if diffs: print(f"      专家分 |API-缓存|: 均值={np.mean(diffs):.3f} 中位数={np.median(diffs):.3f}")
            paras = jload(os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json"))
            rows_api = []
            for iid in sample:
                if iid not in expert_api: continue
                es = expert_api[iid]; s3 = [es.get(sk) for sk in POOL]
                if any(v is None for v in s3): continue
                sp = float(np.std(s3))
                fp = imgs.get(iid)
                if not fp or not os.path.exists(fp): continue
                img = Image.open(fp)
                if ds=="spaq": img.thumbnail((1568,1568), Image.BICUBIC)
                f_raw = opencv_features(img)
                f = np.array([f_raw[k] for k in FEAT_KEYS[:6]]+[sp], dtype=float)
                for j,k in enumerate(FEAT_KEYS):
                    if k in ("lap_var","noise","colorful"): f[j]=np.log(max(f[j],1e-6))
                if ds=="koniq": g_ = softmax(W @ ((f-mu)/sd)); fus = float(g_ @ np.array(s3))
                else: fus = float(np.mean(s3))
                r1b = load_pred(os.path.join(cfg.runs_dir,"final",f"r1b_{ds}","scores.csv"))
                vote = vote_mean(iid, r1b, paras)
                if vote is None: continue
                rows_api.append({"img_id":iid,"score":round(alpha*fus+(1-alpha)*vote,4)})
            if rows_api:
                ma = compute_metrics({r_["img_id"]:r_["score"] for r_ in rows_api}, mos)
                print(f"      API 管线 ({len(rows_api)}张): SRCC={ma['SRCC']:.4f} MAE={ma['MAE']:.4f}")
                cp = load_pred(os.path.join(cfg.runs_dir,"final",r2_dir.replace("r2_","r6_"),"scores.csv"))
                if ds=="spaq":
                    cup = os.path.join(cfg.runs_dir,"posthoc","r6_unified_spaq","scores.csv")
                    if os.path.exists(cup): cp = load_pred(cup)
                common = sorted(set(r_["img_id"] for r_ in rows_api)&set(cp))
                if common:
                    cm = compute_metrics({i:cp[i] for i in common}, mos)
                    pm2 = compute_metrics({r["img_id"]:r["score"] for r in rows_api if r["img_id"] in common}, mos)
                    print(f"      同批图缓存管线: SRCC={cm['SRCC']:.4f} MAE={cm['MAE']:.4f}")
                    print(f"      API vs 缓存: SRCC {pm2['SRCC']-cm['SRCC']:+.4f} MAE {pm2['MAE']-cm['MAE']:+.4f}")
        print(f"\n  账本: {client.ledger()}")
    asyncio.run(run())


def sec_html(cfg):
    print("\n" + "=" * 72)
    print("[HTML] 生成自包含报告")
    print("=" * 72)
    table = json.load(open(TMP, encoding="utf-8")) if os.path.exists(TMP) else []
    html_path = os.path.join(BASE, "verify_report.html")
    import base64
    rows_h = ""
    for row in table:
        if row[1] is None:
            rows_h += f"<tr><td>{row[0]}</td><td colspan='4' class='miss'>数据缺失</td></tr>"
        else:
            rows_h += (f"<tr><td>{row[0]}</td><td>{row[1]:.4f}</td><td>{row[2]:.4f}</td>"
                       f"<td>{row[3]:.4f}</td><td>{row[4]:.4f}</td><td class='src'>{row[5]}</td></tr>")
    figs_b64 = {}
    for name in ["fig4_scatter.png","fig3_w.png"]:
        fp = os.path.join(BASE,"docs","figs",name)
        if os.path.exists(fp):
            with open(fp,"rb") as f: figs_b64[name] = base64.b64encode(f.read()).decode()
    img_tags = ""
    if "fig4_scatter.png" in figs_b64:
        img_tags += f"<img src='data:image/png;base64,{figs_b64['fig4_scatter.png']}' alt='scatter'>"
    if "fig3_w.png" in figs_b64:
        img_tags += f"<img src='data:image/png;base64,{figs_b64['fig3_w.png']}' alt='W matrix'>"

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>IQA Agent 复现验证报告</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#1a1a1a;background:#fcfcfb}}
h1{{font-size:24px;border-bottom:2px solid #1b2a4a;padding-bottom:8px}}h2{{font-size:18px;color:#1b2a4a;margin-top:32px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{padding:8px 12px;text-align:center;border-bottom:1px solid #e1e0d9;font-size:14px}}
th{{background:#1b2a4a;color:#fff}}tr:nth-child(even){{background:#f5f5f5}}
.miss{{color:#d03b3b}}.src{{color:#757575;font-size:12px}}.note{{color:#757575;font-size:13px}}img{{max-width:100%;margin:12px 0}}</style></head><body>
<h1>IQA Agent 框架 — 复现验证报告</h1>
<p class="note">{time.strftime('%Y-%m-%d %H:%M:%S')} | Backbone: Qwen3-VL-32B | 全部数据来自 scores.csv</p>
<h2>主表</h2><table><tr><th>臂</th><th>KonIQ SRCC</th><th>KonIQ MAE</th><th>SPAQ SRCC</th><th>SPAQ MAE</th><th>来源</th></tr>{rows_h}</table>
<h2>散点图 + 门控矩阵</h2>{img_tags}
<p class="note" style="margin-top:40px">本报告由 python scripts/verify.py --html 自动生成。</p></body></html>"""
    with open(html_path,"w",encoding="utf-8") as f: f.write(html)
    print(f"  HTML 报告: {html_path}")


# ═══════════════ main ═══════════════
def main():
    ap = argparse.ArgumentParser(description="IQA Agent 复现验证")
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api-only", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    default = not args.api_only and not args.html_only
    do_api  = args.api or args.api_only
    do_html = args.html or args.html_only

    cfg = get_config()

    if args.html_only:
        if os.path.exists(TMP):
            print(f"加载缓存的表数据: {TMP}")
            sec_html(cfg)
        else:
            print("错误: 未找到 .verify_table.json，请先运行 python scripts/verify.py")
        return

    if args.api_only:
        sec5_api(cfg, args.limit)
        return

    if default:
        sec1_selfcheck(cfg)
        sec2_table(cfg)
        sec3_pipeline(cfg, args.limit)
        sec4_cke(cfg)

    if do_api:
        sec5_api(cfg, args.limit)

    if do_html:
        sec_html(cfg)


if __name__ == "__main__":
    main()
