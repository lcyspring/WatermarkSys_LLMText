"""文本质量代理指标（可离线计算）。

说明：BERTScore / Perplexity 需要模型与网络，本模块先实现
无模型的确定性代理指标，用于 W10 质量评估的预演与回归测试；
真实 BERTScore 在依赖可用时由 ``bertscore_guard`` 调用。
"""
from __future__ import annotations

import math
import zlib
from collections import Counter


def distinct_n(text: str, n: int = 2) -> float:
    """distinct-n：不重复 n-gram 数 / 总 n-gram 数（越高越多样）。"""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(text: str, n: int = 4) -> float:
    """重复 n-gram 占比（越低越好）。"""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    rep = sum(v - 1 for v in c.values() if v > 1)
    return rep / len(grams)


def compression_ratio(text: str) -> float:
    """zlib 压缩比（原长 / 压缩长），越高越冗余。"""
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    comp = zlib.compress(raw, 9)
    return len(raw) / max(1, len(comp))


def avg_token_length(text: str) -> float:
    toks = text.split()
    return sum(len(t) for t in toks) / len(toks) if toks else 0.0


def quality_report(texts: list[str]) -> dict:
    """对一组文本汇总质量代理指标（均值 + 标准差）。"""
    if not texts:
        return {"count": 0}
    rows = []
    for t in texts:
        rows.append({
            "len": len(t.split()),
            "distinct1": distinct_n(t, 1),
            "distinct2": distinct_n(t, 2),
            "repeat4": repetition_rate(t, 4),
            "compression": compression_ratio(t),
        })

    def agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in rows]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        return round(mu, 4), round(sd, 4)

    out = {"count": len(rows)}
    for k in ("len", "distinct1", "distinct2", "repeat4", "compression"):
        mu, sd = agg(k)
        out[f"{k}_mean"] = mu
        out[f"{k}_sd"] = sd
    return out


def bertscore_guard(cands: list[str], refs: list[str]):
    """BERTScore 可选调用；依赖缺失时返回 None（不阻塞主流程）。"""
    try:
        from bert_score import score  # type: ignore
    except Exception:
        return None
    try:
        P, R, F = score(cands, refs, lang="en", verbose=False)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F.mean()),
        }
    except Exception:
        return None
