"""批量生成（W5/W7 交付物）。

用法：
  python scripts/04_generate.py --mode synthetic --deltas 0,0.5,1,2
  python scripts/04_generate.py --mode hf --deltas 0,2 --limit 20 --max-new-tokens 128
"""
from __future__ import annotations

import argparse

from common import GEN_PATH, ensure_dirs, get_config, load_key, read_jsonl
from wm.pipeline.generate import GeneratePipeline, build_adapter, load_prompts as _lp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "hf"])
    ap.add_argument("--deltas", default="0,0.5,1,2")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=str(GEN_PATH))
    ap.add_argument("--fresh", action="store_true", help="覆盖而非追加")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    prompts = _lp(str(__import__("common").PROMPTS))
    if not prompts:
        raise SystemExit("提示集为空，请先运行 scripts/03_make_prompts.py")
    if args.limit:
        prompts = prompts[: args.limit]

    key = load_key()
    mode = args.mode
    vocab = args.vocab
    adapter = build_adapter(mode, vocab_size=vocab, model_id=args.model_id,
                            dtype="fp32", device="cpu")
    if mode == "hf":
        info = adapter.load()
        vocab = info.vocab_size
        print(f"[模型] {info.model_id} vocab={info.vocab_size} device={info.device}")

    pipe = GeneratePipeline(adapter, key, args.out)
    first = True
    for d in [float(x) for x in args.deltas.split(",") if x.strip()]:
        cfg = get_config(delta=d, vocab_size=vocab, mode=mode,
                         seed=args.seed, max_new_tokens=args.max_new_tokens,
                         model_id=args.model_id)
        stats = pipe.run(prompts, cfg, append=not (args.fresh and first))
        first = False
        print(f"[生成] delta={d:<4} ok={stats['ok']} fail={stats['fail']} "
              f"→ {stats['path']}")
    total = len(read_jsonl(args.out))
    print(f"[完成] 累计 {total} 条记录")


if __name__ == "__main__":
    main()
