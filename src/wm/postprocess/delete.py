"""随机删词：以词为单位删除指定比例，记录实际删除率与处理后长度。"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class EditResult:
    text: str
    tokens: list[str]
    original_count: int
    edited_count: int
    actual_rate: float
    failed: bool = False
    error: str = ""


def delete_words(
    text: str,
    rate: float,
    seed: int = 0,
    tokenizer=None,
) -> EditResult:
    """删除约 rate 比例的词；返回结果与实际编辑比例。

    tokenizer 为 None 时按空白分词（记录口径）；正式实验应使用与生成一致的
    tokenizer 重新分词，并记录处理后长度。
    """
    if tokenizer is not None:
        tokens = tokenizer.tokenize(text)
    else:
        tokens = text.split()
    original = len(tokens)
    if original == 0:
        return EditResult(text, [], 0, 0, 0.0, failed=True, error="empty text")
    n_remove = int(round(rate * original))
    n_remove = max(0, min(n_remove, original))
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(original), n_remove), reverse=True)
    kept = list(tokens)
    for i in idxs:
        kept.pop(i)
    out = " ".join(kept) if tokenizer is None else tokenizer.convert_tokens_to_string(kept)
    actual = n_remove / original
    return EditResult(out, kept, original, len(kept), actual)
