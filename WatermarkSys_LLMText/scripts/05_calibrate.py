"""阈值校准（W8 交付物）。

流程：生成 400 条 delta=0 负例（40 提示 × 10 随机种子）→ 逐条检测取 z →
按目标 FPR=1% 取经验分位 → 保存阈值文件（含 Wilson 置信区间）。
"""
from __future__ import annotations

import argparse

from common import (OUT, THRESHOLD_PATH, ensure_dirs, get_config, load_key,
                    write_json, write_jsonl)
from wm.detect.calibrate import calibrate, save_threshold
from wm.pipeline.generate import GeneratePipeline, build_adapter, detect_record
from wm.pipeline.generate import load_prompts as _lp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--mode", default="synthetic")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    ensure_dirs()
    key = load_key()
    import common
    prompts = _lp(str(common.PROMPTS))[: args.prompts]
    adapter = build_adapter(args.mode, vocab_size=args.vocab)

    path = OUT / "calib_negatives.jsonl"
    pipe = GeneratePipeline(adapter, key, str(path))
    total = 0
    for s in range(1000, 1000 + args.seeds):
        cfg = get_config(delta=0.0, vocab_size=args.vocab, mode=args.mode,
                         seed=s, max_new_tokens=args.max_new_tokens)
        stats = pipe.run(prompts, cfg, append=(s != 1000))
        total += stats["ok"]
    print(f"[负例] 生成 {total} 条 → {path}")

    recs = [r for r in __import__("common").read_jsonl(path)
            if r.get("status") == "completed"]
    zs, details = [], []
    cfg = get_config(delta=0.0, vocab_size=args.vocab, mode=args.mode)
    for r in recs:
        det = detect_record(r, cfg, key, threshold=None)
        if det["status"] == "completed":
            zs.append(det["z"])
            details.append({**{k: r[k] for k in ("gen_id", "prompt_id", "delta")},
                            **{k: det[k] for k in ("n", "g", "z", "p", "status")}})
    write_jsonl(details, OUT / "calib_negatives_detections.jsonl")

    result = calibrate(zs, target_fpr=args.target_fpr, threshold_version="v1")
    save_threshold(result, str(THRESHOLD_PATH))
    write_json(result.to_dict(), OUT / "calibration_summary.json")

    print(f"[校准] 有效负例 {len(zs)} 条")
    print(f"  threshold(z) = {result.threshold:.4f}")
    print(f"  经验 FPR = {result.emp_fpr:.4f} "
          f"({sum(1 for z in zs if z >= result.threshold)}/{len(zs)})")
    print(f"  95% CI  = [{result.fpr_ci[0]:.4f}, {result.fpr_ci[1]:.4f}]")
    print(f"  → {THRESHOLD_PATH}")

    from common import REPORTS, markdown_table
    md = [
        "# 阈值校准报告（W8）", "",
        f"- 负例来源：delta=0（生成端完全不偏置），{args.prompts} 提示 × "
        f"{args.seeds} 随机种子",
        f"- 有效负例数：{len(zs)}",
        f"- 目标 FPR：{args.target_fpr:.0%}",
        f"- 选定阈值：**z ≥ {result.threshold:.4f}**（{result.threshold_id}）",
        f"- 校准集经验 FPR：{result.emp_fpr:.4f}",
        f"- Wilson 95% CI：[{result.fpr_ci[0]:.4f}, {result.fpr_ci[1]:.4f}]",
        "",
        "## 关键结论边界", "",
        "1. 阈值只由负例确定，不使用任何正例信息；",
        "2. 三个 delta 档共用同一阈值，避免“按结果调阈值”；",
        "3. 零误报不能证明总体 FPR=0，置信区间上界才是可报告口径；",
        "4. 未检出 ≠ 人类撰写，检出 ≠ 内容真实。",
        "",
        "## 负例 z 分布", "",
    ]
    zs_sorted = sorted(zs)
    md.append(markdown_table(
        ["统计量", "数值"],
        [["样本数", len(zs_sorted)],
         ["最小 z", f"{zs_sorted[0]:.3f}"],
         ["中位数 z", f"{zs_sorted[len(zs_sorted)//2]:.3f}"],
         ["最大 z", f"{zs_sorted[-1]:.3f}"],
         ["阈值 z", f"{result.threshold:.3f}"]]))
    (REPORTS / "阈值校准报告.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[文档] {REPORTS / '阈值校准报告.md'}")


if __name__ == "__main__":
    main()
