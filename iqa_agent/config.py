# -*- coding: utf-8 -*-
"""全局配置：所有超参一处管理。"""
import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv():
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Config:
    # ---- 模型 ----
    model_main: str = "qwen3-vl-32b-instruct"   # 正式跑分
    model_debug: str = "qwen3-vl-8b-instruct"   # 联调/冒烟
    # ---- API ----
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    concurrency: int = 16
    max_retries: int = 8
    timeout: int = 180
    max_tokens: int = 300
    # ---- 数据路径 ----
    koniq_img_dir: str = os.path.join(ROOT, "评测数据集", "koniq-10k", "512x384")
    koniq_val_csv: str = os.path.join(ROOT, "评测数据集", "koniq-10k", "koniq10k_val.csv")
    koniq_train_csv: str = os.path.join(ROOT, "评测数据集", "koniq-10k", "koniq10k_train.csv")
    spaq_img_dir: str = os.path.join(ROOT, "评测数据集", "SPAQ", "images", "TestImage")
    spaq_test_csv: str = os.path.join(ROOT, "评测数据集", "SPAQ", "spaqTest.csv")
    # ---- 输出 ----
    runs_dir: str = os.path.join(ROOT, "runs")
    cache_dir: str = os.path.join(ROOT, "runs", "cache")
    ladder_dir: str = os.path.join(ROOT, "runs", "ladder")
    # ---- 评分量程（D4：原生尺度，预先声明并冻结）----
    scales: dict = field(default_factory=lambda: {"koniq": (1.0, 5.0), "spaq": (0.0, 10.0)})
    # ---- 抽样 ----
    seed: int = 42
    ladder_seed: int = 42     # 阶梯源图抽样
    workset_seed: int = 43    # CKE 工作集抽样（与阶梯不同种子 → 不重叠）
    ladder_n_sources: int = 200
    workset_size: int = 1000


def get_config() -> Config:
    _load_dotenv()
    cfg = Config()
    os.makedirs(cfg.runs_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return cfg
