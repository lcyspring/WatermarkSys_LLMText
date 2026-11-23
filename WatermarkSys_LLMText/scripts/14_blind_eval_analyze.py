"""盲评结果分析（W14 配套）。

读取一名或多名评分者填好的 rating 文件，输出：

  1. 每个评分维度上“有水印 − 无水印”的配对均值差与 95% 置信区间；
  2. 评分者间一致性 ICC(2,1)；
  3. 方向一致率（两名评分者是否都认为同一侧更好）。

用法：
  python scripts/14_blind_eval_analyze.py --ratings outputs/blind_eval/rating_r1.csv ^
                                          --ratings outputs/blind_eval/rating_r2.csv

评分文件列：pair_id, fluency_A, fluency_B, coherence_A, coherence_B
（留空的单元格会被跳过；未填满时脚本会提示完成度。）
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from common import OUT, REPORTS, ensure_dirs

TASKS = ("fluency", "coherence")


def load_ratings(path: Path) -> dict[str, dict[str, float]]:
    """返回 {pair_id: {列名: 分数}}，只保留已填写的数值。"""
    rows: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for rec in csv.DictReader(f):
            pid = (rec.get("pair_id") or "").strip()
            if not pid:
                continue
            vals: dict[str, float] = {}
            for key, raw in rec.items():
                if key == "pair_id" or raw is None:
                    continue
                raw = str(raw).strip()
                if not raw:
                    continue
                try:
                    vals[key] = float(raw)
                except ValueError:
                    continue
            rows[pid] = vals
    return rows


def load_key(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def icc21(matrix: np.ndarray) -> float:
    """ICC(2,1)：双向随机效应、单评分者。matrix 形状 (目标数, 评分者数)。"""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((matrix - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / denom if denom > 0 else float("nan")


def analyze(raters: list[dict[str, dict[str, float]]],
            key: dict[str, dict[str, str]]) -> tuple[list[str], dict]:
    n_raters = len(raters)
    pids = sorted(set.intersection(*(set(r) for r in raters))) if raters else []
    lines: list[str] = []
    summary: dict = {"pairs": len(pids), "raters": n_raters}

    filled = 0
    total_cells = 0
    for r in raters:
        for pid, vals in r.items():
            total_cells += len(TASKS) * 2
            filled += sum(1 for k in vals if k.endswith(("_A", "_B")))
    completeness = filled / total_cells if total_cells else 0.0

    lines.append("# 盲评分析报告\n")
    lines.append(f"- 评分者数量：{n_raters}")
    lines.append(f"- 参与配对数：{len(pids)}")
    lines.append(f"- 评分完成度：{completeness:.1%}\n")

    diffs_by_task: dict[str, list[float]] = {t: [] for t in TASKS}
    for task in TASKS:
        lines.append(f"## {task}：有水印 − 无水印（配对）\n")
        lines.append("| 评分者 | 有效对数 | 均值差 | 95% CI |")
        lines.append("|---|---|---|---|")
        for i, r in enumerate(raters, 1):
            ds = []
            for pid in pids:
                vals = r.get(pid, {})
                a, b = vals.get(f"{task}_A"), vals.get(f"{task}_B")
                if a is None or b is None or pid not in key:
                    continue
                wm = a if key[pid]["A_label"] == "watermarked" else b
                bl = b if key[pid]["A_label"] == "watermarked" else a
                ds.append(wm - bl)
            if len(ds) >= 2:
                mean = float(np.mean(ds))
                half = 1.96 * float(np.std(ds, ddof=1)) / math.sqrt(len(ds))
                lines.append(f"| R{i} | {len(ds)} | {mean:+.3f} | "
                             f"[{mean - half:+.3f}, {mean + half:+.3f}] |")
                diffs_by_task[task].extend(ds)
        lines.append("")

    # 评分者间一致性：目标 = 每个 pair 的 A 侧与 B 侧
    lines.append("## 评分者间一致性 ICC(2,1)\n")
    lines.append("| 维度 | ICC(2,1) | 说明 |")
    lines.append("|---|---|---|")
    for task in TASKS:
        targets, columns = [], []
        for pid in pids:
            for side in ("A", "B"):
                row = [r.get(pid, {}).get(f"{task}_{side}") for r in raters]
                if any(v is None for v in row):
                    continue
                targets.append(row)
        if targets:
            mat = np.array(targets, dtype=float)
            value = icc21(mat)
            note = "一致性良好" if value >= 0.75 else ("中等" if value >= 0.5 else "偏低")
            lines.append(f"| {task} | {value:.3f} | {note}（n={mat.shape[0]}） |")
        else:
            lines.append(f"| {task} | — | 无完整双评分数据 |")
    lines.append("")

    summary["completeness"] = completeness
    summary["mean_diff"] = {t: (float(np.mean(v)) if v else float("nan"))
                            for t, v in diffs_by_task.items()}
    return lines, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", action="append", required=True,
                    help="评分文件路径，可重复传入多份（每份代表一位评分者）")
    ap.add_argument("--key", default=str(OUT / "blind_eval" / "key.csv"))
    ap.add_argument("--out", default=str(REPORTS / "盲评分析报告.md"))
    args = ap.parse_args()

    ensure_dirs()
    raters = [load_ratings(Path(p)) for p in args.ratings]
    key = load_key(Path(args.key))
    lines, summary = analyze(raters, key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[盲评分析] 评分者={summary['raters']} 配对数={summary['pairs']} "
          f"完成度={summary['completeness']:.1%}")
    for task, diff in summary["mean_diff"].items():
        print(f"  {task}: 均值差 {diff:+.3f}")
    print(f"  报告: {out_path}")


if __name__ == "__main__":
    main()
