# -*- coding: utf-8 -*-
"""异步 VLM 客户端：并发 + 指数退避重试 + SHA256 磁盘缓存 + token 账本。

缓存键 = sha256(model + messages 完整内容)，任何调用可断点续跑。
"""
import asyncio
import base64
import hashlib
import json
import os
import time

from openai import AsyncOpenAI, APIConnectionError, APIError, APITimeoutError, RateLimitError

from .config import Config


class VLMClient:
    def __init__(self, cfg: Config, model: str):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        assert api_key and api_key.startswith("sk-"), "DASHSCOPE_API_KEY 未配置（检查 .env）"
        self.cfg = cfg
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=cfg.base_url, timeout=cfg.timeout)
        self._sem = asyncio.Semaphore(cfg.concurrency)
        # 账本
        self.calls = 0
        self.cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0

    # ---------- 缓存 ----------
    def _cache_path(self, key: str) -> str:
        shard = key[:2]
        d = os.path.join(self.cfg.cache_dir, shard)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, key + ".json")

    def _key(self, messages, temperature: float, max_tokens: int) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        # 生成参数必须进 key（F-008：裁判温度不同的调用不能共享缓存）。
        # 为兼容既有缓存：temp=0 且 max_tokens=默认值 的评分调用维持原 key 格式。
        if temperature != 0 or max_tokens != self.cfg.max_tokens:
            h.update(f"|T={temperature}|MT={max_tokens}".encode())
        h.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode())
        return h.hexdigest()

    # ---------- 图像编码 ----------
    @staticmethod
    def image_uri(path: str) -> str:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        fmt = "jpeg" if ext in ("jpg", "jpeg") else ext
        size = os.path.getsize(path)
        if size <= 9_500_000:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/{fmt};base64,{b64}"
        # 超 API 上传上限（10MB）：保持分辨率不变，仅降 JPEG 质量重新编码（F-011）
        import io
        from PIL import Image
        img = Image.open(path).convert("RGB")
        for q in (85, 75, 65):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q)
            if buf.tell() <= 9_500_000:
                break
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    # ---------- 核心调用 ----------
    async def chat(self, messages, temperature: float = 0.0, max_tokens: int | None = None):
        """返回 (text, usage_dict)。带缓存与重试。"""
        max_tokens = max_tokens or self.cfg.max_tokens
        key = self._key(messages, temperature, max_tokens)
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            self.cache_hits += 1
            return obj["text"], obj["usage"]

        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                async with self._sem:
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                text = resp.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"text": text, "usage": usage}, f, ensure_ascii=False)
                self.calls += 1
                self.tokens_in += usage["prompt_tokens"]
                self.tokens_out += usage["completion_tokens"]
                return text, usage
            except RateLimitError as e:
                last_err = e
                await asyncio.sleep(min(5 * (2 ** attempt), 60))
            except (APITimeoutError, APIConnectionError) as e:
                last_err = e
                await asyncio.sleep(3 * (2 ** attempt))
            except APIError as e:
                last_err = e
                if getattr(e, "status_code", 500) and 400 <= e.status_code < 500 and e.status_code != 429:
                    raise  # 4xx（非限流）不重试，直接抛
                await asyncio.sleep(2 * (2 ** attempt))
        raise RuntimeError(f"API 调用失败，重试 {self.cfg.max_retries} 次仍失败: {last_err}")

    async def score_image(self, image_path: str, prompt: str, temperature: float = 0.0):
        """单图评分便捷入口：文本 + 图像的 user 消息。"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self.image_uri(image_path)}},
            ],
        }]
        return await self.chat(messages, temperature=temperature)

    async def compare_images(self, path_a: str, path_b: str, prompt: str, temperature: float = 0.0):
        """双图对比便捷入口（C 路线 pairwise）。"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self.image_uri(path_a)}},
                {"type": "image_url", "image_url": {"url": self.image_uri(path_b)}},
            ],
        }]
        return await self.chat(messages, temperature=temperature)

    def ledger(self) -> dict:
        return {
            "model": self.model,
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


async def gather_with_progress(tasks, every: int = 100, label: str = ""):
    """并发执行并打印进度。单个任务失败不拖垮整批：异常存入结果列表由调用方统计。"""
    results = [None] * len(tasks)
    done = 0
    t0 = time.time()

    async def _wrap(i, coro):
        nonlocal done
        try:
            results[i] = await coro
        except Exception as e:  # noqa: BLE001 - 批处理容错，由调用方决定如何处理
            results[i] = e
        done += 1
        if done % every == 0 or done == len(tasks):
            dt = time.time() - t0
            rate = done / max(dt, 1e-9)
            print(f"  [{label}] {done}/{len(tasks)}  ({dt:.0f}s, ~{rate:.1f}/s)", flush=True)

    await asyncio.gather(*[_wrap(i, c) for i, c in enumerate(tasks)])
    n_err = sum(1 for r in results if isinstance(r, Exception))
    if n_err:
        print(f"  [{label}] ⚠️ {n_err} 个任务失败（缓存已保留成功部分，重跑可补齐）", flush=True)
    return results
