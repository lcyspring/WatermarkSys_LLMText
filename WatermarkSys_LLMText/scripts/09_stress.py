"""压力与一致性测试（W6 D6 / W13 交付物）。

四项：
1. 错误密钥：用水印文本 + 错误密钥检测，z 应回到 0 附近（密钥特异性）；
2. gamma 不一致：用 0.20 / 0.30 检测 gamma=0.25 生成的文本，z 应下降；
3. 温度敏感性：同一 delta 在 temperature 0.7/1.0/1.3 下的 z；
4. 配置一致性：比较不同配置快照的 fingerprint，确保换参即换 hash。
"""
from __future__ import annotations

import argparse

from common import (OUT, REPORTS, ensure_dirs, get_config, load_key,
                    markdown_table, read_jsonl, write_json)
from wm.pipeline.generate import GeneratePipeline, build_adapter, detect_record
from wm.watermark import detect as kgw_detect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    ensure_dirs()
    key = load_key()
    wrong = load_key(wrong=True)
    import common
    prompts = common.load_prompts(str(common.PROMPTS))[: args.limit]
    adapter = build_adapter("synthetic", vocab_size=args.vocab)

    cfg2 = get_config(delta=2.0, vocab_size=args.vocab,
                      max_new_tokens=args.max_new_tokens)
    pipe = GeneratePipeline(adapter, key, str(OUT / "stress_gens.jsonl"))
    pipe.run(prompts, cfg2, append=False)
    gens = [r for r in read_jsonl(OUT / "stress_gens.jsonl")
            if r.get("status") == "completed"]

    results = {"key_test": {}, "gamma_test": {}, "temperature_test": {},
               "consistency": {}}

    # 1. 密钥正确 vs 错误
    for name, k in (("正确密钥", key), ("错误密钥", wrong)):
        zs = []
        for g in gens:
            ids = list(g["prompt_token_ids"]) + list(g["token_ids"])
            r = kgw_detect(ids, key=k, gamma=cfg2.gamma,
                           context_width=cfg2.context_width,
                           vocab_size=cfg2.vocab_size,
                           ngram_block=cfg2.ngram_block,
                           score_from=g["prompt_len"])
            if r["status"] == "completed":
                zs.append(r["z"])
        results["key_test"][name] = round(sum(zs) / len(zs), 3) if zs else None

    # 2. gamma 不一致
    for gm in (0.20, 0.25, 0.30):
        zs = []
        for g in gens:
            ids = list(g["prompt_token_ids"]) + list(g["token_ids"])
            r = kgw_detect(ids, key=key, gamma=gm,
                           context_width=cfg2.context_width,
                           vocab_size=cfg2.vocab_size,
                           ngram_block=cfg2.ngram_block,
                           score_from=g["prompt_len"])
            if r["status"] == "completed":
                zs.append(r["z"])
        results["gamma_test"][str(gm)] = round(sum(zs) / len(zs), 3) if zs else None

    # 3. 温度敏感性
    for temp in (0.7, 1.0, 1.3):
        cfg = get_config(delta=2.0, vocab_size=args.vocab, temperature=temp,
                         max_new_tokens=args.max_new_tokens)
        out = OUT / f"stress_temp_{temp}.jsonl"
        GeneratePipeline(adapter, key, str(out)).run(prompts, cfg, append=False)
        zs = []
        for g in read_jsonl(out):
            if g.get("status") != "completed":
                continue
            d = detect_record(g, cfg, key, threshold=None)
            if d["status"] == "completed":
                zs.append(d["z"])
        results["temperature_test"][str(temp)] = (
            round(sum(zs) / len(zs), 3) if zs else None)

    # 4. 配置一致性
    pairs = {
        "delta=0 vs delta=2": (get_config(delta=0.0, vocab_size=args.vocab),
                               get_config(delta=2.0, vocab_size=args.vocab)),
        "gamma=0.25 vs 0.30": (
            get_config(delta=2.0, vocab_size=args.vocab),
            __import__("wm.config", fromlist=["WatermarkConfig"]).WatermarkConfig(
                delta=2.0, vocab_size=args.vocab, gamma=0.30)),
        "h=4 vs h=5": (
            get_config(delta=2.0, vocab_size=args.vocab),
            __import__("wm.config", fromlist=["WatermarkConfig"]).WatermarkConfig(
                delta=2.0, vocab_size=args.vocab, context_width=5)),
    }
    for label, (a, b) in pairs.items():
        results["consistency"][label] = {
            "hash_a": a.config_hash(), "hash_b": b.config_hash(),
            "same_fingerprint": a.fingerprint() == b.fingerprint(),
            "different_hash": a.config_hash() != b.config_hash(),
        }

    write_json(results, OUT / "stress_results.json")

    lines = [
        "# 压力与一致性测试报告（W6/W13）", "",
        "## 1. 密钥特异性", "",
        markdown_table(["检测密钥", "平均 z"],
                       [[k, v] for k, v in results["key_test"].items()]),
        "",
        "> 错误密钥下 z 应回到 0 附近，说明水印由密钥决定，无法被无密钥者读出。",
        "",
        "## 2. gamma 必须一致", "",
        markdown_table(["检测用 gamma", "平均 z"],
                       [[k, v] for k, v in results["gamma_test"].items()]),
        "",
        "## 3. 温度敏感性（delta=2）", "",
        markdown_table(["temperature", "平均 z"],
                       [[k, v] for k, v in results["temperature_test"].items()]),
        "",
        "## 4. 配置一致性检查", "",
        markdown_table(["配置对比", "hash_a", "hash_b", "指纹相同", "hash 不同"],
                       [[k, v["hash_a"], v["hash_b"],
                         v["same_fingerprint"], v["different_hash"]]
                        for k, v in results["consistency"].items()]),
        "",
        "> 任一参数变化都必须改变 config_hash，否则结论将无法回溯。",
    ]
    (REPORTS / "压力与一致性测试报告.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    print("[压力测试] 完成")
    print("  密钥:", results["key_test"])
    print("  gamma:", results["gamma_test"])
    print("  温度:", results["temperature_test"])
    print(f"[文档] {REPORTS / '压力与一致性测试报告.md'}")


if __name__ == "__main__":
    main()
