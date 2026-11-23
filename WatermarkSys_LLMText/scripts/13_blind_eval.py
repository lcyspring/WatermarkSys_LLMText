"""盲评材料生成（W14 交付物）。

从生成数据中抽取“有水印 vs 无水印”成对样本，随机化左右位置，
产出三份文件供人工盲评：

  outputs/blind_eval/pairs.csv    只含 pair_id 与 A/B 文本（不含标签）
  outputs/blind_eval/key.csv      标签映射（盲评结束后再打开）
  outputs/blind_eval/rating.csv   评分表模板（流畅度/连贯性 1—5 分）

盲评口径：评分者只看 pairs.csv，不接触 key.csv；至少 2 人独立评分，
用平均值与一致性（如 Cohen's kappa / ICC）报告。
"""
from __future__ import annotations

import csv
import os
import random

from common import GEN_PATH, OUT, ensure_dirs, read_jsonl

TASKS = ("fluency", "coherence")


def main() -> None:
    ensure_dirs()
    gens = [g for g in read_jsonl(GEN_PATH) if g.get("status") == "completed"]
    by_key = {(g["prompt_id"], float(g["delta"])): g for g in gens}
    prompts = sorted({g["prompt_id"] for g in gens})

    rng = random.Random(20261123)
    pairs, keys, ratings = [], [], []
    idx = 0
    for pid in prompts:
        base = by_key.get((pid, 0.0))
        if not base:
            continue
        for d in (1.0, 2.0):
            wm = by_key.get((pid, d))
            if not wm:
                continue
            idx += 1
            pair_id = f"B{idx:04d}"
            wm_left = rng.random() < 0.5
            a_text, b_text = ((wm["text"], base["text"]) if wm_left
                              else (base["text"], wm["text"]))
            pairs.append({"pair_id": pair_id,
                          "text_A": a_text, "text_B": b_text,
                          "length": len(wm["text"].split())})
            keys.append({"pair_id": pair_id,
                         "A_label": "watermarked" if wm_left else "baseline",
                         "B_label": "baseline" if wm_left else "watermarked",
                         "delta": d})
            row = {"pair_id": pair_id}
            for t in TASKS:
                row[f"{t}_A"] = ""
                row[f"{t}_B"] = ""
            ratings.append(row)

    out_dir = OUT / "blind_eval"
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in (("pairs.csv", pairs), ("key.csv", keys),
                       ("rating.csv", ratings)):
        with open(out_dir / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"[盲评] 生成 {len(pairs)} 对样本（目标 120 对）")
    print(f"  盲评材料: {out_dir / 'pairs.csv'}")
    print(f"  标签映射: {out_dir / 'key.csv'}（盲评结束后再打开）")
    print(f"  评分模板: {out_dir / 'rating.csv'}")
    print("\n  评分口径：fluency / coherence 各 1—5 分；")
    print("  报告口径：均值差 + 至少 2 人评分的 ICC / kappa。")


if __name__ == "__main__":
    main()
