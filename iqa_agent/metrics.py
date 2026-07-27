# -*- coding: utf-8 -*-
"""评测指标：SRCC / MAE / PLCC。

⚠️ 合规：本模块是全项目唯二接触 MOS 的代码之一（另一个是 50_eval.py/70_figures.py）。
仅在最终评测与考后分析阶段调用（ADR-0001）。
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .data import load_mos  # noqa: F401  （唯一合法 import 点，便于 grep 审计）


def compute_metrics(pred: dict[str, float], mos: dict[str, float]) -> dict:
    """pred/mos: {img_id: score}。仅在共同键上计算。"""
    keys = [k for k in pred if k in mos and pred[k] is not None]
    if len(keys) < 3:
        return {"n": len(keys), "SRCC": None, "MAE": None, "PLCC": None}
    p = np.array([pred[k] for k in keys], dtype=float)
    m = np.array([mos[k] for k in keys], dtype=float)
    srcc = spearmanr(p, m).statistic
    plcc = pearsonr(p, m).statistic
    mae = float(np.mean(np.abs(p - m)))
    return {"n": len(keys), "SRCC": round(float(srcc), 4),
            "MAE": round(mae, 4), "PLCC": round(float(plcc), 4)}
