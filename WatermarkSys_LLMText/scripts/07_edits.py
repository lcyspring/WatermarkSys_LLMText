"""后处理扰动实验（W9 交付物）。

对被水印文本（delta ∈ {1,2}）施加 3 类 × 3 档扰动：
  删词 10%/20%/30%、截断 75%/50%/25%、替换 10%/20%/30%
共 60×2×9 = 1080 个变体，逐条统计“扰动后仍可检出”的比例。
"""
from __future__ import annotations

import argparse

from common import (EDIT_PATH, GEN_PATH, OUT, ensure_dirs, get_config,
                    load_key, load_threshold, read_jsonl, write_jsonl)
from wm.postprocess.token_edits import EDIT_LEVELS, apply_edit
from wm.watermark import detect as kgw_detect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="1,2")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", default=str(OUT / "edit_detections.jsonl"))
    args = ap.parse_args()

    ensure_dirs()
    key = load_key()
    tau, tau_id = load_threshold()
    gens = [r for r in read_jsonl(GEN_PATH)
            if r.get("status") == "completed"
            and float(r.get("delta", 0)) in
            [float(x) for x in args.deltas.split(",")]]
    print(f"[输入] 待扰动生成 {len(gens)} 条")

    cfg = get_config(vocab_size=args.vocab)
    rows, variants = [], []
    for g in gens:
        base_ids = list(g.get("prompt_token_ids", []))
        plen = int(g.get("prompt_len", 0))
        new_ids = list(g["token_ids"])
        for kind, levels in EDIT_LEVELS.items():
            for lv in levels:
                res = apply_edit(new_ids, kind, lv, args.vocab, seed=len(rows))
                ids = base_ids + res.tokens
                det = kgw_detect(
                    ids, key=key, gamma=cfg.gamma,
                    context_width=cfg.context_width, vocab_size=cfg.vocab_size,
                    ngram_block=cfg.ngram_block, threshold=tau,
                    min_tokens=32, score_from=plen,
                    key_id=cfg.key_id, threshold_id=tau_id)
                row = {
                    "gen_id": g["gen_id"], "prompt_id": g["prompt_id"],
                    "delta": g["delta"], "edit_type": kind,
                    "level": lv, "target_rate": lv,
                    "actual_rate": round(res.actual_rate, 4),
                    "len_before": res.original_count,
                    "len_after": res.edited_count,
                    "status": det["status"], "n": det["n"], "g": det["g"],
                    "z": det["z"], "p": det["p"],
                    "decision": det["decision"],
                    "detected": bool(det["decision"]),
                }
                rows.append(row)
                variants.append({
                    "gen_id": g["gen_id"], "type": kind,
                    "target_rate": lv, "actual_rate": res.actual_rate,
                    "length_before": res.original_count,
                    "length_after": res.edited_count,
                    "status": res.status,
                    "text": " ".join(str(t) for t in res.tokens),
                })
    write_jsonl(rows, args.out)
    write_jsonl(variants, EDIT_PATH)

    print(f"[扰动] 变体 {len(rows)} 条 → {args.out}")
    print("\n  类型      档位   变体数  有效n均值  检出率")
    for kind in EDIT_LEVELS:
        for lv in EDIT_LEVELS[kind]:
            sub = [r for r in rows if r["edit_type"] == kind and r["level"] == lv]
            ok = [r for r in sub if r["status"] == "completed"]
            rate = (sum(1 for r in ok if r["decision"]) / len(ok)) if ok else 0.0
            nmean = (sum(r["n"] for r in ok) / len(ok)) if ok else 0.0
            print(f"  {kind:<9} {lv:<6} {len(sub):<7} {nmean:<10.1f} {rate:.1%}")


if __name__ == "__main__":
    main()
