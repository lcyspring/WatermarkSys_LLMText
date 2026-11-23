"""指标汇总与图表数据（W12 交付物）。

输出：
  outputs/metrics/main_results.csv
  outputs/metrics/length_strata.csv
  outputs/metrics/robustness.csv
  docs/reports/主结果报告.md
  docs/reports/鲁棒性报告.md
"""
from __future__ import annotations

import argparse
import csv
import os

from common import (DET_PATH, METRICS_PATH, OUT, REPORTS, ensure_dirs,
                    load_threshold, markdown_table, read_jsonl, write_json)
from wm.eval import (BinaryMetrics, confusion_at_threshold, length_bin,
                     roc_auc, stratified_by_length, summarize_z,
                     tpr_at_fpr)


def write_csv(rows: list[dict], path) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default=str(DET_PATH))
    ap.add_argument("--edits", default=str(OUT / "edit_detections.jsonl"))
    ap.add_argument("--vocab", type=int, default=2048)
    args = ap.parse_args()

    ensure_dirs()
    tau, tau_id = load_threshold()
    det = [d for d in read_jsonl(args.detections)
           if d.get("status") == "completed"]
    if not det:
        raise SystemExit("没有可用检测结果，请先运行 06_detect.py")

    deltas = sorted({float(d["delta"]) for d in det})
    neg = [d for d in det if float(d["delta"]) == 0.0]
    main_rows, summary = [], {"threshold": tau, "threshold_id": tau_id,
                              "by_delta": {}}

    for dl in deltas:
        sub = [d for d in det if float(d["delta"]) == dl]
        if dl == 0.0:
            m = confusion_at_threshold([d["z"] for d in sub],
                                       [False] * len(sub), tau)
            row = {"delta": dl, "role": "negative",
                   "n": len(sub),
                   "z_mean": round(sum(d["z"] for d in sub) / len(sub), 3),
                   "detected_rate": m.fpr,
                   "fpr": round(m.fpr, 4), "fnr": "-",
                   "auc": "-", "tpr_at_fpr1": "-"}
        else:
            zs = [d["z"] for d in sub] + [d["z"] for d in neg]
            labels = [True] * len(sub) + [False] * len(neg)
            m = confusion_at_threshold(zs, labels, tau)
            auc = roc_auc(zs, labels)
            _, tpr1 = tpr_at_fpr(zs, labels, 0.01)
            row = {"delta": dl, "role": "watermarked",
                   "n": len(sub),
                   "z_mean": round(sum(d["z"] for d in sub) / len(sub), 3),
                   "detected_rate": round(sum(1 for d in sub if d["decision"]) / len(sub), 4),
                   "fpr": round(m.fpr, 4), "fnr": round(m.fnr, 4),
                   "auc": round(auc, 4), "tpr_at_fpr1": round(tpr1, 4)}
        main_rows.append(row)
        summary["by_delta"][str(dl)] = {**row,
                                        "z_dist": summarize_z(sub)}

    write_csv(main_rows, OUT / "metrics" / "main_results.csv")

    # 长度分层（delta 最大档）
    top = max(deltas)
    top_rows = [d for d in det if float(d["delta"]) == top]
    strata_rows = []
    for b, recs in stratified_by_length(top_rows).items():
        ok = [r for r in recs if r["status"] == "completed"]
        if not ok:
            continue
        strata_rows.append({
            "length_bin": b, "n_samples": len(ok),
            "z_mean": round(sum(r["z"] for r in ok) / len(ok), 3),
            "detected_rate": round(
                sum(1 for r in ok if r["decision"]) / len(ok), 4),
        })
    write_csv(strata_rows, OUT / "metrics" / "length_strata.csv")

    # 鲁棒性
    edits = read_jsonl(args.edits)
    rob_rows = []
    if edits:
        for kind in sorted({e["edit_type"] for e in edits}):
            for lv in sorted({e["level"] for e in edits if e["edit_type"] == kind}):
                sub = [e for e in edits
                       if e["edit_type"] == kind and e["level"] == lv
                       and e.get("status") == "completed"]
                if not sub:
                    continue
                rob_rows.append({
                    "edit_type": kind, "level": lv,
                    "n_variants": len(sub),
                    "n_mean": round(sum(e["n"] for e in sub) / len(sub), 1),
                    "detected_rate": round(
                        sum(1 for e in sub if e["decision"]) / len(sub), 4),
                    "z_mean": round(sum(e["z"] for e in sub) / len(sub), 3),
                    "min_n": min(e["n"] for e in sub),
                })
        write_csv(rob_rows, OUT / "metrics" / "robustness.csv")

    summary["robustness"] = rob_rows
    summary["length_strata"] = strata_rows
    summary["n_detections"] = len(det)
    write_json(summary, METRICS_PATH)

    # ---------------- 文档 ----------------
    lines = [
        "# 主结果报告（W12）", "",
        f"- 阈值：z ≥ {tau:.4f}（{tau_id}，仅由负例校准）",
        f"- 检测记录数：{len(det)}",
        f"- 负例（delta=0）：{len(neg)} 条", "",
        "## 各 delta 档表现", "",
        markdown_table(
            ["delta", "角色", "样本数", "平均 z", "判定为有水印比例",
             "FPR(全负例)", "FNR", "AUC", "TPR@FPR=1%"],
            [[r["delta"], r["role"], r["n"], r["z_mean"],
              f"{r['detected_rate']:.2%}" if isinstance(r["detected_rate"], float) else r["detected_rate"],
              r["fpr"], r["fnr"], r["auc"], r["tpr_at_fpr1"]]
             for r in main_rows]),
        "",
        "## 长度分层（最高 delta 档）", "",
        markdown_table(["有效长度区间", "样本数", "平均 z", "检出率"],
                       [[s["length_bin"], s["n_samples"], s["z_mean"],
                         f"{s['detected_rate']:.2%}"] for s in strata_rows]),
        "",
        "## 判读要点", "",
        f"1. delta=0 的判定率即校准后 FPR，应贴近 1%（实测 "
        f"{main_rows[0]['detected_rate']:.2%}）。",
        "2. delta 越大，平均 z 与检出率单调上升，说明偏置机制方向正确。",
        "3. 短文本（32–63 token）的检出率通常最低，这与统计功效随 n 增长一致。",
        "4. 未检出 ≠ 人类撰写：检测只回答“是否检出指定配置的水印证据”。",
        "",
        "> 数据模式：离线机制级模拟（SyntheticLM，V=2048）。真实模型结果需",
        "> 在有 GPU 与权重的环境中以 --mode hf 重跑，统计口径与脚本完全一致。",
    ]
    (REPORTS / "主结果报告.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")

    if rob_rows:
        rl = [
            "# 鲁棒性报告（后处理扰动，W9/W13）", "",
            f"- 阈值：z ≥ {tau:.4f}",
            "- 扰动对象：delta ∈ {1,2} 的含水印文本，共 "
            f"{len({e['gen_id'] for e in edits})} 条源文本",
            "", "## 各类扰动下的检出率", "",
            markdown_table(["扰动类型", "档位", "变体数", "平均有效 n",
                            "平均 z", "检出率"],
                           [[r["edit_type"], r["level"], r["n_variants"],
                             r["n_mean"], r["z_mean"],
                             f"{r['detected_rate']:.2%}"] for r in rob_rows]),
            "",
            "## 解读", "",
            "1. 删词与截断直接减少有效 token 数 n，检测功效随之下降；",
            "2. 截断 25% 时 n 可能跌破 32 的下限，状态会转为 insufficient_length；",
            "3. 替换类扰动保留长度但破坏 n-gram，z 下降幅度应与替换率正相关；",
            "4. 结论边界：以上为机制级模拟下的趋势，真实模型需复测。",
            "",
        ]
        (REPORTS / "鲁棒性报告.md").write_text("\n".join(rl) + "\n",
                                              encoding="utf-8")

    print("[指标] 阈值", tau)
    for r in main_rows:
        print(f"  delta={r['delta']:<4} n={r['n']:<4} z_mean={r['z_mean']:<7} "
              f"检出率={r['detected_rate']}")
    print(f"[文件] {OUT/'metrics'}; 文档 → {REPORTS}")


if __name__ == "__main__":
    main()
