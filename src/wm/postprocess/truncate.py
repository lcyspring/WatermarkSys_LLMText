"""截断：固定保留前缀，作为长度消融实验。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TruncateResult:
    text: str
    original_count: int
    kept_count: int
    kept_ratio: float
    failed: bool = False
    error: str = ""


def truncate(text: str, keep_ratio: float, tokenizer=None) -> TruncateResult:
    """保留文本前缀的 keep_ratio 比例。keep_ratio 应在 (0, 1]。"""
    if not (0.0 < keep_ratio <= 1.0):
        return TruncateResult(text, 0, 0, keep_ratio, failed=True,
                              error="keep_ratio out of (0,1]")
    if tokenizer is not None:
        tokens = tokenizer.tokenize(text)
    else:
        tokens = text.split()
    original = len(tokens)
    if original == 0:
        return TruncateResult(text, 0, 0, keep_ratio, failed=True, error="empty text")
    keep = max(1, int(round(keep_ratio * original)))
    keep = min(keep, original)
    kept = tokens[:keep]
    out = " ".join(kept) if tokenizer is None else tokenizer.convert_tokens_to_string(kept)
    return TruncateResult(out, original, len(kept), len(kept) / original)
