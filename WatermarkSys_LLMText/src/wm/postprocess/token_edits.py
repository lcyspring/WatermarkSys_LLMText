"""token 级后处理：直接在 token 序列上模拟删词 / 截断 / 替换。

为什么需要 token 级实现：检测以 token 为单位统计绿名单命中，
真实文本后处理（字符串编辑）后需重新分词；token 级模拟用于
断网预演，保证“同一条生成 → 多种扰动 → 同一检测器”的口径一致。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class TokenEditResult:
    tokens: list[int] = field(default_factory=list)
    original_count: int = 0
    edited_count: int = 0
    target_rate: float = 0.0
    actual_rate: float = 0.0
    status: str = "completed"
    error: str = ""


def delete_random(tokens: list[int], rate: float, seed: int = 0) -> TokenEditResult:
    original = len(tokens)
    if original == 0:
        return TokenEditResult([], 0, 0, rate, 0.0, "failed", "empty")
    n_remove = max(0, min(int(round(rate * original)), original))
    rng = random.Random(seed)
    drop = set(rng.sample(range(original), n_remove))
    kept = [t for i, t in enumerate(tokens) if i not in drop]
    return TokenEditResult(kept, original, len(kept), rate,
                           n_remove / original)


def truncate(tokens: list[int], keep_ratio: float) -> TokenEditResult:
    original = len(tokens)
    if original == 0:
        return TokenEditResult([], 0, 0, keep_ratio, 0.0, "failed", "empty")
    keep = max(1, min(int(round(keep_ratio * original)), original))
    kept = tokens[:keep]
    return TokenEditResult(kept, original, len(kept), keep_ratio,
                           keep / original)


def substitute_random(tokens: list[int], rate: float, vocab_size: int,
                      seed: int = 0) -> TokenEditResult:
    """随机替换为其他 token，模拟同义改写 / 回译造成的表层扰动。"""
    original = len(tokens)
    if original == 0:
        return TokenEditResult([], 0, 0, rate, 0.0, "failed", "empty")
    n_sub = max(0, min(int(round(rate * original)), original))
    rng = random.Random(seed)
    idxs = rng.sample(range(original), n_sub)
    out = list(tokens)
    for i in idxs:
        new = rng.randrange(max(2, vocab_size))
        while new == out[i]:
            new = rng.randrange(max(2, vocab_size))
        out[i] = new
    return TokenEditResult(out, original, len(out), rate, n_sub / original)


EDIT_LEVELS = {
    "delete": [0.10, 0.20, 0.30],
    "truncate": [0.75, 0.50, 0.25],
    "substitute": [0.10, 0.20, 0.30],
}


def apply_edit(tokens: list[int], kind: str, level: float, vocab_size: int,
               seed: int = 0) -> TokenEditResult:
    if kind == "delete":
        return delete_random(tokens, level, seed)
    if kind == "truncate":
        return truncate(tokens, level)
    if kind == "substitute":
        return substitute_random(tokens, level, vocab_size, seed)
    raise ValueError(f"unknown edit kind: {kind}")
