"""批量检测（W6/W12 交付物）：对生成记录逐条检测并输出 z/p/n/判定。"""
from __future__ import annotations

import argparse
import os

from common import (GEN_PATH, DET_PATH, ensure_dirs, get_config, load_key,
                    load_threshold, read_jsonl, write_jsonl)
from wm.db import connect, init_db, insert_detection
from wm.pipeline.generate import detect_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(GEN_PATH))
    ap.add_argument("--out", default=str(DET_PATH))
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--db", default=str(__import__("common").DB_PATH))
    args = ap.parse_args()

    ensure_dirs()
    if not os.path.exists(args.input):
        raise SystemExit(f"缺少生成文件 {args.input}，请先运行 04_generate.py")

    key = load_key()
    if args.threshold is not None:
        tau, tau_id = args.threshold, "tau-manual"
    else:
        tau, tau_id = load_threshold()

    records = [r for r in read_jsonl(args.input) if r.get("status") == "completed"]
    cfg = get_config(delta=0.0, vocab_size=args.vocab)
    out = []
    for r in records:
        det = detect_record(r, cfg, key, threshold=tau, threshold_id=tau_id)
        det["config_hash"] = r.get("config_hash")
        det["prompt_len"] = r.get("prompt_len")
        det["det_id"] = f"d-{r['gen_id']}"
        det["source"] = "generation"
        out.append(det)
    write_jsonl(out, args.out)

    init_db(args.db)
    conn = connect(args.db)
    try:
        for d in out:
            insert_detection(conn, d)
        conn.commit()
    finally:
        conn.close()

    by_delta = {}
    for d in out:
        by_delta.setdefault(d["delta"], []).append(d)
    print(f"[检测] {len(out)} 条 → {args.out}")
    print(f"  阈值 {tau:.3f} ({tau_id})  数据库 {args.db}")
    for dl in sorted(by_delta):
        rows = [x for x in by_delta[dl] if x["status"] == "completed"]
        if not rows:
            continue
        pos = sum(1 for x in rows if x["decision"])
        zs = sum(x["z"] for x in rows) / len(rows)
        print(f"  delta={dl:<4} n={len(rows):<4} 判为有水印={pos:<4} "
              f"({pos/len(rows):.1%})  平均 z={zs:.2f}")


if __name__ == "__main__":
    main()
