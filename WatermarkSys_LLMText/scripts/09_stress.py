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

    # 4. 重复 n-gram 屏蔽消融（ngram_block=0 vs 4）
    #    4a. 正常生成文本：随机性高，屏蔽几乎不触发
    #    4b. 构造“复读文本”：同一 4-gram 反复出现，屏蔽可防止 z 被虚高
    ablation = {}
    for nb in (0, 4):
        zs, ns = [], []
        for g in gens:
            ids = list(g["prompt_token_ids"]) + list(g["token_ids"])
            r = kgw_detect(ids, key=key, gamma=cfg2.gamma,
                           context_width=cfg2.context_width,
                           vocab_size=cfg2.vocab_size,
                           ngram_block=nb,
                           score_from=g["prompt_len"])
            if r["status"] == "completed":
                zs.append(r["z"])
                ns.append(r["n"])
        ablation[f"正常文本 / ngram_block={nb}"] = {
            "mean_z": round(sum(zs) / len(zs), 3) if zs else None,
            "n": round(sum(ns) / len(ns), 1) if ns else None,
            "masked": 0,
        }
    # 4b. 构造“复读文本”：取一个在自身上下文下恒为绿名单的 token 反复输出，
    #     此时不屏蔽会把同一段复读反复计入 n 与 g，z 被极度虚高。
    h = cfg2.context_width
    V = cfg2.vocab_size
    from wm.watermark.kgw import green_list
    rep_tok = next(t for t in range(V)
                   if t in green_list(key, [t] * h, h, cfg2.gamma, V))
    rep_ids = [rep_tok] * (h + 240)
    for nb in (0, 4):
        r = kgw_detect(rep_ids, key=key, gamma=cfg2.gamma,
                       context_width=h, vocab_size=V,
                       ngram_block=nb, score_from=h)
        ablation[f"复读文本 / ngram_block={nb}"] = {
            "mean_z": round(r["z"], 3), "n": r["n"],
            "masked": len(r.get("masked", [])),
            "status": r["status"],
        }
    results["ngram_ablation"] = ablation

    # 5. 配置一致性
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
        "## 4. 重复 n-gram 屏蔽消融", "",
        markdown_table(["设置", "指标", "数值"],
                       [[k, "平均 z / z", v["mean_z"]] for k, v in
                        results["ngram_ablation"].items()]
                       + [[k, "有效 n", v["n"]] for k, v in
                          results["ngram_ablation"].items()]
                       + [[k, "屏蔽位置数", v["masked"]] for k, v in
                          results["ngram_ablation"].items()
                          if v["masked"]]),
        "",
        "> 正常随机文本几乎不产生重复 4-gram，屏蔽开关对结果无影响；",
        "> 而“复读文本”下若不屏蔽，重复片段会反复计入 n 与 g，导致 z 虚高、",
        "> 把机器复读误判为水印证据。这是默认开启 ngram_block=4 的原因。",
        "",
        "## 5. 配置一致性检查", "",
        markdown_table(["配置对比", "hash_a", "hash_b", "指纹相同", "hash 不同"],
                       [[k, v["hash_a"], v["hash_b"],
                         v["same_fingerprint"], v["different_hash"]]
                        for k, v in results["consistency"].items()]),
        "",
        "> 任一参数变化都必须改变 config_hash，否则结论将无法回溯。",
        "> delta 属实验变量，不进入算法指纹（同一水印方案），但必须进入 hash。",
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
