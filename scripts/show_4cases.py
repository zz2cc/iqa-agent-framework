# -*- coding: utf-8 -*-
"""4张真实图(KonIQ×2 + SPAQ×2)完整走统一框架，展示全部prompt和output。"""
import csv, json, os, sys, numpy as np
sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_images
from iqa_agent.prompts.skills import build_skill_prompt, _SCALE_BLOCK
import importlib.util

cfg = get_config()
POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
B4 = {
    "koniq": [
        "Rate the overall quality of this image on a scale from 1 to 5. Reply with only a single number.",
        "On a scale from 1 to 5, how would you rate the overall quality of this image? Reply with only a single number.",
        "Give a single overall quality score for this image, from 1 (worst) to 5 (best). Reply with only the number.",
        "As an image quality rater, assign one overall quality score from 1 to 5 to this image. Reply with only a single number.",
    ],
    "spaq": [
        "Rate the overall quality of this image on a scale from 0 to 10. Reply with only a single number.",
        "On a scale from 0 to 10, how would you rate the overall quality of this image? Reply with only a single number.",
        "Give a single overall quality score for this image, from 0 (worst) to 10 (best). Reply with only the number.",
        "As an image quality rater, assign one overall quality score from 0 to 10 to this image. Reply with only a single number.",
    ],
}


def opencv_features(img):
    from PIL import Image
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    c = gray[1:-1, 1:-1]
    lap = -4 * c + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    lap_var = float(np.var(lap))
    box = (gray[:-2, :-2] + gray[:-2, 1:-1] + gray[:-2, 2:] + gray[1:-1, :-2] + c +
           gray[1:-1, 2:] + gray[2:, :-2] + gray[2:, 1:-1] + gray[2:, 2:]) / 9.0
    res = c - box; gx = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2])
    thr = np.quantile(gx, 0.25); flat = res[gx <= thr]
    noise = float(1.4826 * np.median(np.abs(flat - np.median(flat)))) if flat.size else 0.0
    rg = arr[..., 0] - arr[..., 1]; yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    h, w = gray.shape
    return {"lap_var": lap_var, "noise": noise, "colorful": colorful,
            "bright": float(gray.mean()), "logpix": float(np.log(h * w)), "aspect": float(w / h)}


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def load_domain(ds, eval_ds):
    images = {r.img_id: r.path for r in load_images(cfg, eval_ds)}
    r2 = {}
    with open(os.path.join(cfg.runs_dir, "final", f"r2_{ds}", "scores.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("skill_scores"): r2[r["img_id"]] = json.loads(r["skill_scores"])
    r1b = {}
    with open(os.path.join(cfg.runs_dir, "final", f"r1b_{ds}", "scores.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("score") not in ("", None): r1b[r["img_id"]] = float(r["score"])
    paras = {}
    pp = os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json")
    if os.path.exists(pp):
        paras = json.load(open(pp, encoding="utf-8"))
    # R6 unified output
    r6u = {}
    r6up = os.path.join(cfg.runs_dir, "posthoc", f"r6_unified_{ds}", "scores.csv")
    if os.path.exists(r6up):
        with open(r6up, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f): r6u[r["img_id"]] = r["score"]
    return images, r2, r1b, paras, r6u


def show_case(label, ds, iid, images, r2, r1b, paras, r6u, gate_model, alpha, lo, hi):
    from PIL import Image
    img = Image.open(images[iid])

    # ── Step 1: OpenCV ──
    feat = opencv_features(img)
    es = r2[iid]
    s3 = [es[sk] for sk in POOL]
    sp = float(np.std(s3))

    # ── Step 2: Expert prompts ──
    for sk in POOL:
        prompt = build_skill_prompt(sk, ds)
        score = s3[POOL.index(sk)]
        print(f"\n{'─'*65}")
        print(f"  [{sk}] 注入给模型的完整Prompt (≈500 tokens):")
        print(f"{'─'*65}")
        for line in prompt.split("\n"):
            print(f"  | {line}")
        print(f"{'─'*65}")
        print(f"  模型输入: 上面这段文字 + {img.size[0]}×{img.size[1]} 像素图")
        print(f"  模型输出: {{\"level\": X, \"score\": {score:.2f}, \"reason\": \"...\"}}")

    # ── Step 3: Dynamic fusion ──
    if gate_model is not None:
        mu_a = np.array(gate_model["mu"])
        sd_a = np.array([s if s > 1e-6 else 1.0 for s in gate_model["sd"]])
        W = np.array(gate_model["W"])
        f = np.array([feat["lap_var"], feat["noise"], feat["colorful"],
                      feat["bright"], feat["logpix"], feat["aspect"], sp], dtype=float)
        for j, k in enumerate(["lap_var","noise","colorful","bright","logpix","aspect","spread"]):
            if k in ("lap_var","noise","colorful"): f[j] = np.log(max(f[j], 1e-6))
        f_std = (f - mu_a) / sd_a
        logits = W @ f_std
        g = softmax(logits)
        fus = float(g @ np.array(s3))
    else:
        g = np.ones(3) / 3
        fus = float(np.mean(s3))
        logits = None

    # ── Step 4: Bare voting ──
    bp = [r1b[iid]] + [paras.get(iid, {}).get(str(k)) for k in (1, 2, 3)]
    vote = float(np.mean(bp))

    final = alpha * fus + (1 - alpha) * vote

    # ── Print everything ──
    print(f"\n{'='*65}")
    print(f"  {label}  ({iid})")
    print(f"  文件: {images[iid]}")
    print(f"  尺寸: {img.size[0]}×{img.size[1]}")
    print(f"{'='*65}")

    print(f"\n  [步骤1] OpenCV特征提取 (纯numpy, 不调API)")
    print(f"    lap_var={feat['lap_var']:.1f}  noise={feat['noise']:.3f}  colorful={feat['colorful']:.1f}")
    print(f"    bright={feat['bright']:.1f}  logpix={feat['logpix']:.2f}  aspect={feat['aspect']:.2f}")
    print(f"    spread(3专家标准差)={sp:.3f}")

    print(f"\n  [步骤2] 3个专家分 (来自R2考试日缓存, 每张图3次API已付)")
    for sk, val in zip(POOL, s3):
        print(f"    {sk:12s} = {val:.2f}")

    print(f"\n  [步骤3] 动态融合 (纯数学, 不调API)")
    if gate_model is not None:
        print(f"    标准化特征 z = [{', '.join(f'{v:+.3f}' for v in f_std)}]")
        print(f"    W(3x7)矩阵(从BT锦标赛学来,已冻结):")
        for i, sk in enumerate(POOL):
            print(f"      {sk:12s}: [{', '.join(f'{w:+.4f}' for w in W[i])}]")
        print(f"    logits = W@z = [{logits[0]:+.3f}, {logits[1]:+.3f}, {logits[2]:+.3f}]")
        print(f"    softmax → TECH={g[0]:.3f} GLOBAL={g[1]:.3f} CONTENT={g[2]:.3f}")
    else:
        print(f"    SPAQ门控FAIL→等权回退: TECH=GLOBAL=CONTENT=1/3")
    print(f"    融合分 = {s3[0]:.2f}×{g[0]:.3f} + {s3[1]:.2f}×{g[1]:.3f} + {s3[2]:.2f}×{g[2]:.3f} = {fus:.4f}")

    print(f"\n  [步骤4] Bare投票 (4种措辞, 无锚点/无专家) ")
    for idx in range(4):
        val = r1b[iid] if idx == 0 else paras.get(iid, {}).get(str(idx), "?")
        print(f"    para{idx}: \"{B4[ds][idx][:70]}...\"")
        print(f"           模型返回: {val}")
    print(f"    4条均值 = {vote:.3f}")

    print(f"\n  [步骤5] 最终融合")
    r6_str = r6u.get(iid, "?")
    print(f"    R6 = {alpha} × {fus:.4f} + {1-alpha} × {vote:.4f}")
    print(f"       = {alpha*fus:.4f} + {(1-alpha)*vote:.4f}")
    print(f"       = {final:.4f}")
    print(f"    统一框架输出: {r6_str}")
    print()


# ── Main ──
images_k, r2k, r1bk, paras_k, r6u_k = load_domain("koniq", "koniq_val")
images_s, r2s, r1bs, paras_s, r6u_s = load_domain("spaq", "spaq_test")

gate_k = json.load(open(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json")))

# Pick 2 KonIQ (low + high) and 2 SPAQ (low + high)
common_k = sorted(set(r2k) & set(r1bk) & set(r6u_k), key=lambda i: float(r6u_k[i]))
common_s = sorted(set(r2s) & set(r1bs) & set(r6u_s), key=lambda i: float(r6u_s[i]))

print("=" * 70)
print("统一框架 — 4张真实图完整链路")
print("=" * 70)

# Show expert prompts first (once per domain, they're identical across images)
print("\n" + "█" * 65)
print("█  S-TECH 专家 Prompt (KonIQ版, 1-5量程) — 3个专家共用此结构")
print("█" * 65)
for line in build_skill_prompt("S-TECH", "koniq").split("\n"):
    print(f"│ {line}")
print("█" * 65)
print("█  S-GLOBAL和S-CONTENT同理, 仅换维度定义/检查清单/文字等级")
print("█" * 65)

print("\n" + "█" * 65)
print("█  S-TECH 专家 Prompt (SPAQ版, 0-10量程) — 仅Score scale段不同")
print("█" * 65)
for line in build_skill_prompt("S-TECH", "spaq").split("\n"):
    print(f"│ {line}")
print("█" * 65)

# Bare prompts
print("\n" + "█" * 65)
print("█  Bare投票 Prompt (4种措辞, 无锚点/无专家/无JSON)")
print("█" * 65)
for idx, t in enumerate(B4["koniq"]):
    print(f"│  para{idx}: \"{t}\"")
print("█" * 65)
print("█  SPAQ版仅换量程: 1→0, 5→10")

# Show 4 cases
show_case("案例1: KonIQ最低分", "koniq", common_k[0], images_k, r2k, r1bk, paras_k, r6u_k, gate_k, 0.6, 1, 5)
show_case("案例2: KonIQ最高分", "koniq", common_k[-1], images_k, r2k, r1bk, paras_k, r6u_k, gate_k, 0.6, 1, 5)
show_case("案例3: SPAQ最低分", "spaq", common_s[0], images_s, r2s, r1bs, paras_s, r6u_s, None, 0.3, 0, 10)
show_case("案例4: SPAQ最高分", "spaq", common_s[-1], images_s, r2s, r1bs, paras_s, r6u_s, None, 0.3, 0, 10)
