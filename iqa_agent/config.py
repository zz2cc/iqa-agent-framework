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
    model_main: str = "qwen3-vl-32b-instruct"
    model_debug: str = "qwen3-vl-8b-instruct"
    # ---- API ----
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    concurrency: int = 16
    max_retries: int = 8
    timeout: int = 180
    max_tokens: int = 300
    # ---- 数据路径 ----
    #  设置环境变量 IQADATA=你的数据集目录。
    #  目录结构: $IQADATA/koniq-10k/512x384/ 和 $IQADATA/SPAQ/images/TestImage/
    koniq_img_dir: str = ""
    koniq_val_csv: str = ""
    koniq_train_csv: str = ""
    spaq_img_dir: str = ""
    spaq_test_csv: str = ""
    # ---- 输出 ----
    runs_dir: str = os.path.join(ROOT, "runs")
    cache_dir: str = os.path.join(ROOT, "runs", "cache")
    ladder_dir: str = os.path.join(ROOT, "runs", "ladder")
    # ---- 评分量程（D4：原生尺度，预先声明并冻结）----
    scales: dict = field(default_factory=lambda: {"koniq": (1.0, 5.0), "spaq": (0.0, 10.0)})
    # ---- 抽样 ----
    seed: int = 42
    ladder_seed: int = 42
    workset_seed: int = 43
    ladder_n_sources: int = 200
    workset_size: int = 1000

    def __post_init__(self):
        _data = os.environ.get("IQADATA", os.path.join(ROOT, "评测数据集"))
        # 路径不存在时，CSV 标签文件回退到项目根目录（仓库自带副本）
        def _data_or_root(subpath):
            p = os.path.join(_data, subpath)
            if os.path.exists(p):
                return p
            return os.path.join(ROOT, os.path.basename(subpath))
        if not self.koniq_img_dir:
            self.koniq_img_dir = os.path.join(_data, "koniq-10k", "512x384")
        if not self.koniq_val_csv:
            self.koniq_val_csv = _data_or_root("koniq-10k/koniq10k_val.csv")
        if not self.koniq_train_csv:
            self.koniq_train_csv = _data_or_root("koniq-10k/koniq10k_train.csv")
        if not self.spaq_img_dir:
            self.spaq_img_dir = os.path.join(_data, "SPAQ", "images", "TestImage")
        if not self.spaq_test_csv:
            self.spaq_test_csv = _data_or_root("SPAQ/spaqTest.csv")


def get_config() -> Config:
    _load_dotenv()
    cfg = Config()
    os.makedirs(cfg.runs_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return cfg
