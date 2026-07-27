# -*- coding: utf-8 -*-
"""
API 冒烟测试：验证 DashScope OpenAI 兼容端点 + qwen3-vl 图像评分全链路。

检查项：
  1. 认证与模型名可用
  2. base64 图像输入被接受
  3. 响应中的分数可被正则解析
  4. 单次调用延迟 / token 消耗（用于全量成本估算）
  5. 微梯子 sanity check：加噪图得分应低于干净图

用法： python scripts/smoke_test.py [--model qwen3-vl-8b-instruct]
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=os.path.join(ROOT, ".env")):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def make_test_images(size=(512, 384)):
    """合成两张图：干净渐变图 + 同图加高斯噪声（微梯子 sanity check）。"""
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    clean = np.stack([x, np.flipud(x), np.full_like(x, 128)], axis=-1).astype(np.uint8)
    rng = np.random.default_rng(42)
    noisy = np.clip(clean.astype(np.float32) + rng.normal(0, 40, clean.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(clean), Image.fromarray(noisy)


def img_to_data_uri(img: Image.Image, fmt="JPEG", quality=90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


# 与正式管线同构的评分 prompt（离散等级法，LLM-IQA 式）
SMOKE_PROMPT = """You are an expert image quality assessor.
Assess the TECHNICAL quality of this image (sharpness, noise, exposure, artifacts).
Choose exactly one quality level:
  5 = Excellent: no visible distortions
  4 = Good: minor distortions, not annoying
  3 = Fair: noticeable distortions
  2 = Poor: severe distortions, annoying
  1 = Bad: extreme distortions, unusable
Reply with JSON only: {"level": <1-5>, "reason": "<one short sentence>"}"""


def parse_score(text: str):
    """从模型输出中提取 1-5 等级分；容忍 markdown 代码块。"""
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "level" in obj:
                return float(obj["level"]), obj.get("reason", "")
        except json.JSONDecodeError:
            pass
    m = re.search(r"level[\"'\s:]+([1-5])\b", text)
    if m:
        return float(m.group(1)), ""
    m = re.search(r"\b([1-5])\s*(?:/5|分)?", text)
    return (float(m.group(1)), "") if m else (None, text[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    args = ap.parse_args()

    load_env()
    key = os.environ.get("DASHSCOPE_API_KEY")
    base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    assert key and key.startswith("sk-"), "DASHSCOPE_API_KEY 未配置"
    print(f"[config] model={args.model}  base_url={base}")

    client = OpenAI(api_key=key, base_url=base, timeout=120)
    clean, noisy = make_test_images()

    results = {}
    total_in = total_out = 0
    for name, img in [("clean", clean), ("noisy", noisy)]:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            temperature=0,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SMOKE_PROMPT},
                    {"type": "image_url", "image_url": {"url": img_to_data_uri(img)}},
                ],
            }],
        )
        dt = time.time() - t0
        text = resp.choices[0].message.content
        usage = resp.usage
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens
        score, reason = parse_score(text)
        results[name] = score
        print(f"\n[{name}] {dt:.1f}s | in={usage.prompt_tokens} out={usage.completion_tokens} tokens")
        print(f"  raw: {text[:200]}")
        print(f"  parsed: level={score} reason={reason[:80]}")

    print("\n========== SMOKE SUMMARY ==========")
    ok_parse = all(v is not None for v in results.values())
    ok_order = results["clean"] is not None and results["noisy"] is not None and results["clean"] >= results["noisy"]
    print(f"  auth/model        : PASS (requests succeeded)")
    print(f"  score parsing     : {'PASS' if ok_parse else 'FAIL'} {results}")
    print(f"  micro-ladder order: {'PASS' if ok_order else 'FAIL'} (clean={results['clean']} >= noisy={results['noisy']})")
    print(f"  avg latency       : measured above per call")
    print(f"  tokens per call   : in≈{total_in // 2}, out≈{total_out // 2}")
    # 粗略成本外推（北京站价格以控制台为准，此处按国际站 8B: $0.072/M in, $0.287/M out）
    cost_1k = (total_in / 2 * 1000) / 1e6 * 0.072 + (total_out / 2 * 1000) / 1e6 * 0.287
    print(f"  est cost per 1k calls (8B): ${cost_1k:.2f}")
    sys.exit(0 if (ok_parse and ok_order) else 1)


if __name__ == "__main__":
    main()
