# -*- coding: utf-8 -*-
"""展示两张真实图的完整推理链路：prompt输入 + 模型输出 + 特征 + 权重"""
import json, csv, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqa_agent.config import get_config
from iqa_agent.data import load_images
from iqa_agent.prompts.skills import build_skill_prompt, build_r1_bare_prompt, _SCALE_BLOCK
import importlib.util

cfg = get_config()

spec = importlib.util.spec_from_file_location("m97",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "97_r6_offanchor.py"))
m97 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m97)

from PIL import Image

# ── 加载数据 ──
def load_all(cfg, ds, split):
    images = {r.img_id: r.path for r in load_images(cfg, split)}
    r2 = {}; r1b = {}; r6 = {}
    r2_path = os.path.join(cfg.runs_dir, "final", f"r2_{ds}", "scores.csv")
    with open(r2_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("skill_scores"):
                r2[r["img_id"]] = json.loads(r["skill_scores"])
    r1b_path = os.path.join(cfg.runs_dir, "final", f"r1b_{ds}", "scores.csv")
    with open(r1b_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("score"):
                r1b[r["img_id"]] = float(r["score"])
    r6_path = os.path.join(cfg.runs_dir, "final", f"r6_{ds}", "scores.csv")
    with open(r6_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r6[r["img_id"]] = {"score": r["score"], "reason": r.get("reason","")}
    paras_path = os.path.join(cfg.runs_dir, f"r6_{ds}_paras.json")
    paras = json.load(open(paras_path)) if os.path.exists(paras_path) else {}
    return images, r2, r1b, r6, paras

images_k, r2k, r1bk, r6k, paras_k = load_all(cfg, "koniq", "koniq_val")
images_s, r2s, r1bs, r6s, paras_s = load_all(cfg, "spaq", "spaq_test")

fusion_k = json.load(open(os.path.join(cfg.runs_dir, "router_v3", "fusion_koniq.json")))
POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]

# ── 挑图：KonIQ最低分 + SPAQ最低分 ──
common_k = sorted(set(r2k) & set(r6k) & set(r1bk), key=lambda i: float(r6k[i]["score"]))
common_s = sorted(set(r2s) & set(r6s) & set(r1bs), key=lambda i: float(r6s[i]["score"]))
k_img_id = common_k[0]
s_img_id = common_s[0]

# ════════════════════════════════════════════
# 图1: KonIQ
# ════════════════════════════════════════════
img = Image.open(images_k[k_img_id])
feat = m97.opencv_features(img)
skills3 = [r2k[k_img_id][sk] for sk in POOL]
spread = float(np.std(skills3))
fus, g = m97.dynamic_fusion(fusion_k, skills3, feat)
vote_parts = [r1bk[k_img_id]] + [paras_k.get(k_img_id,{}).get(str(k)) for k in (1,2,3)]
vote = float(np.mean(vote_parts))
final = 0.6*fus + 0.4*vote

print("=" * 70)
print(f"图1: KonIQ  {k_img_id}")
print(f"文件: {images_k[k_img_id]}    尺寸: {img.size[0]}x{img.size[1]}")
print("=" * 70)

print("\n── 步骤1: OpenCV特征(纯numpy, 不调API) ──")
print(f"  输入: {img.size[0]}x{img.size[1]}像素 → numpy数组")
print(f"  输出: lap_var={feat['lap_var']:.1f}  noise={feat['noise']:.3f}  "
      f"colorful={feat['colorful']:.1f}  bright={feat['bright']:.1f}  "
      f"logpix={feat['logpix']:.2f}  aspect={feat['aspect']:.2f}  spread={spread:.3f}")

# Prompt A: S-TECH
print("\n── 步骤2之前提: 3个专家分(来自R2考试日缓存) ──")
print(f"  S-TECH   = {skills3[0]:.2f}")
print(f"  S-GLOBAL = {skills3[1]:.2f}")
print(f"  S-CONTENT= {skills3[2]:.2f}")
print(f"  spread   = {spread:.3f}")
print()
print("  【S-TECH专家注入给模型的完整prompt(≈500 tokens)】:")
print("  ╔══════════════════════════════════════════════════════╗")
for line in build_skill_prompt("S-TECH", "koniq").split("\n"):
    print(f"  ║ {line[:66]:<66} ║")
print("  ╚══════════════════════════════════════════════════════╝")
print(f"  模型收到: 上面这段文字 + {img.size[0]}x{img.size[1]}像素图")
print(f"  模型返回: {{\"level\": X, \"score\": {skills3[0]:.2f}, \"reason\": \"...\"}}")
print(f"  (S-GLOBAL和S-CONTENT同理, 仅换维度定义/检查清单/文字等级)")

print("\n── 步骤2: 动态融合(纯数学, W×z→softmax) ──")
# 手动算一遍
import numpy as np
f = np.array([feat[k] for k in fusion_k["feat_keys"][:-1]] + [0.0], dtype=float)
f[6] = spread
for j, k in enumerate(fusion_k["feat_keys"]):
    if k in ("lap_var", "noise", "colorful"):
        f[j] = np.log(max(f[j], 1e-6))
f = (f - np.array(fusion_k["mu"])) / np.array([s if s > 1e-6 else 1.0 for s in fusion_k["sd"]])
print(f"  z(标准化后) = [{', '.join(f'{x:+.3f}' for x in f)}]")
logits = np.array(fusion_k["W"]) @ f
print(f"  logits = W @ z = [{logits[0]:+.3f}, {logits[1]:+.3f}, {logits[2]:+.3f}]")
e = np.exp(logits - logits.max())
print(f"  softmax → TECH={g[0]:.3f}  GLOBAL={g[1]:.3f}  CONTENT={g[2]:.3f}")
print(f"  融合分 = {skills3[0]:.2f}x{g[0]:.3f} + {skills3[1]:.2f}x{g[1]:.3f} + {skills3[2]:.2f}x{g[2]:.3f} = {fus:.3f}")

print("\n── 步骤3之前: bare投票(4种措辞) ──")
for idx, t in enumerate([
    f"Rate the overall quality of this image on a scale from 1 to 5. Reply with only a single number.",
    f"On a scale from 1 to 5, how would you rate the overall quality of this image? Reply with only a single number.",
    f"Give a single overall quality score for this image, from 1 (worst) to 5 (best). Reply with only the number.",
    f"As an image quality rater, assign one overall quality score from 1 to 5 to this image. Reply with only a single number.",
]):
    val = r1bk[k_img_id] if idx == 0 else paras_k.get(k_img_id,{}).get(str(idx))
    print(f"  para{idx} prompt: \"{t[:80]}...\"")
    print(f"         模型返回: {val}")
print(f"  4条均值 = {vote:.3f}")
print(f"  注意: 这4条prompt不含任何锚点表/专家人设/检查清单")

print("\n── 步骤3: 最终融合 ──")
print(f"  R6 = 0.6 x {fus:.3f} + 0.4 x {vote:.3f} = {final:.4f}")

# ════════════════════════════════════════════
# 图2: SPAQ
# ════════════════════════════════════════════
print("\n\n")
img = Image.open(images_s[s_img_id])
feat = m97.opencv_features(img)
skills3 = [r2s[s_img_id][sk] for sk in POOL]
spread = float(np.std(skills3))
vote_parts = [r1bs[s_img_id]] + [paras_s.get(s_img_id,{}).get(str(k)) for k in (1,2,3)]
vote = float(np.mean(vote_parts))

print("=" * 70)
print(f"图2: SPAQ  {s_img_id}")
print(f"文件: {images_s[s_img_id]}    尺寸: {img.size[0]}x{img.size[1]}")
print("=" * 70)

print(f"\n── 步骤1: OpenCV特征 ──")
print(f"  lap_var={feat['lap_var']:.1f}  noise={feat['noise']:.3f}  "
      f"colorful={feat['colorful']:.1f}  bright={feat['bright']:.1f}  "
      f"logpix={feat['logpix']:.2f}  aspect={feat['aspect']:.2f}  spread={spread:.3f}")

print(f"\n── 3个专家分 ──")
print(f"  S-TECH   = {skills3[0]:.2f}")
print(f"  S-GLOBAL = {skills3[1]:.2f}")
print(f"  S-CONTENT= {skills3[2]:.2f}")
print(f"  spread   = {spread:.3f}")
print(f"\n  【S-TECH注入prompt——与KonIQ完全相同,仅Score scale段换成SPAQ版】:")
print(f"  ╔══════════════════════════════════════════════════════╗")
for line in build_skill_prompt("S-TECH", "spaq").split("\n"):
    print(f"  ║ {line[:66]:<66} ║")
print(f"  ╚══════════════════════════════════════════════════════╝")
print(f"  模型返回: {{\"level\": X, \"score\": {skills3[0]:.2f}, \"reason\": \"...\"}}")

print(f"\n── bare投票(量程0-10) ──")
for idx, t in enumerate([
    f"Rate the overall quality of this image on a scale from 0 to 10. Reply with only a single number.",
    f"On a scale from 0 to 10, how would you rate the overall quality of this image? Reply with only a single number.",
    f"Give a single overall quality score for this image, from 0 (worst) to 10 (best). Reply with only the number.",
    f"As an image quality rater, assign one overall quality score from 0 to 10 to this image. Reply with only a single number.",
]):
    val = r1bs[s_img_id] if idx == 0 else paras_s.get(s_img_id,{}).get(str(idx))
    print(f"  para{idx}: \"{t[:80]}...\" → {val}")
print(f"  4条均值 = {vote:.3f}")

print(f"\n── 门控矩阵 ──")
print(f"  SPAQ侧3x7门控矩阵待训练（统一框架步骤1）。")
print(f"  当前旧R6-SPAQ分（非统一框架）= {r6s[s_img_id]['score']}")
print(f"  统一后R6-SPAQ = 0.6 x dynamic_fusion + 0.4 x bare投票")
print(f"  其中 bare投票 = {vote:.3f}, 动态融合 = 待门控训练后确定")
