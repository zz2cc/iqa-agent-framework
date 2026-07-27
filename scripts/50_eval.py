# -*- coding: utf-8 -*-
"""最终评测（S7）——全项目唯一读取评测集 MOS 的脚本。

用法：
  # 全部路线跑完后，一次性评测：
  python scripts/50_eval.py --runs runs/final

扫描 --runs 目录下所有含 scores.csv 的子目录，按 dataset 关联 MOS，输出：
  runs/final/main_table.csv   （route × dataset × SRCC/MAE/PLCC）
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iqa_agent.config import get_config
from iqa_agent.data import load_mos
from iqa_agent.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="含各路线 scores.csv 的目录")
    args = ap.parse_args()

    cfg = get_config()
    mos_cache = {}

    def mos_for(ds):
        if ds not in mos_cache:
            mos_cache[ds] = load_mos(cfg, ds)
        return mos_cache[ds]

    table = []
    for sub in sorted(os.listdir(args.runs)):
        scores_path = os.path.join(args.runs, sub, "scores.csv")
        if not os.path.isfile(scores_path):
            continue
        with open(scores_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        dataset = rows[0]["dataset"]
        eval_ds = "koniq_val" if dataset == "koniq" else "spaq_test"
        route = rows[0]["route"]
        pred = {r["img_id"]: (float(r["score"]) if r["score"] not in ("", None, "None") else None)
                for r in rows}
        metrics = compute_metrics(pred, mos_for(eval_ds))
        table.append({"run": sub, "route": route, "dataset": eval_ds, **metrics})
        print(f"{sub:<45} {route:<5} {eval_ds:<10} SRCC={metrics['SRCC']} MAE={metrics['MAE']} PLCC={metrics['PLCC']} n={metrics['n']}")

    out_path = os.path.join(args.runs, "main_table.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "route", "dataset", "n", "SRCC", "MAE", "PLCC"])
        w.writeheader()
        w.writerows(table)
    print(f"\n[done] 主表 → {out_path}")
    print("[frozen] 冻结线：此后主表不许重算，修改只进 post-hoc 分支。")


if __name__ == "__main__":
    main()
