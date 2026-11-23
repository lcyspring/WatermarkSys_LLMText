"""KGW 绿名单水印核心算法。

设计要点：
- 纯 Python + hashlib/stdlib 实现绿名单划分与检测统计量，不依赖 torch，
  便于在任何环境（含 CI、断网）下做算法正确性单测；
- 生成端的 logits 偏置（KGWLogitsProcessor）在调用时才导入 torch，
  保证本模块在无 torch 环境下仍可被导入与测试核心逻辑；
- 所有随机划分由 HMAC-SHA256(key, context) 派生，跨平台确定性一致。
"""
from __future__ import annotations

import hashlib
import hmac
import math
import random
import struct
from typing import Iterable, List, Sequence, Tuple

# 检测状态机（与开题报告/任务书口径一致）
STATUS_PLANNED = "planned"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INSUFFICIENT_LENGTH = "insufficient_length"
STATUS_CONFIG_MISMATCH = "config_mismatch"

DEFAULT_MIN_TOKENS = 32


# --------------------------------------------------------------------------- #
# 伪随机划分
# --------------------------------------------------------------------------- #
def _context_seed(
    key: bytes,
    context_tokens: Sequence[int],
    context_width: int,
    algo_version: str,
) -> bytes:
    """由密钥 + 上下文派生确定性种子（HMAC-SHA256）。"""
    msg = algo_version.encode("utf-8")
    msg += struct.pack("<I", int(context_width))
    for tok in context_tokens[-context_width:]:
        msg += struct.pack("<q", int(tok))
    return hmac.new(key, msg, hashlib.sha256).digest()


def green_list(
    key: bytes,
    context_tokens: Sequence[int],
    context_width: int,
    gamma: float,
    vocab_size: int,
    algo_version: str = "kgw-v1",
) -> set:
    """返回当前位置的绿名单 token id 集合。

    取前 ``gamma * vocab_size`` 个（按算法规则取整）伪随机置换位置。
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0, 1)")
    ctx = tuple(int(t) for t in context_tokens[-context_width:])
    seed = _context_seed(key, ctx, context_width, algo_version)
    rng = random.Random(seed)
    perm = list(range(int(vocab_size)))
    rng.shuffle(perm)
    k = max(1, int(round(gamma * vocab_size)))
    return set(perm[:k])


# --------------------------------------------------------------------------- #
# 检测统计量
# --------------------------------------------------------------------------- #
def count_green_hits(
    tokens: Sequence[int],
    key: bytes,
    gamma: float,
    context_width: int,
    vocab_size: int,
    ngram_block: int = 4,
    score_from: int | None = None,
) -> Tuple[int, int, List[int]]:
    """统计有效位置数 n 与绿名单命中数 g。

    - 只对 ``score_from``（默认 = context_width）之后的位置计分，排除提示前缀；
    - 若某位置会形成已出现过的重复 n-gram，则屏蔽该位置（不计入 n、g），
      避免模型自我强化导致统计量偏置；
    - 返回 (n, g, masked_positions)。
    """
    start = context_width if score_from is None else int(score_from)
    tokens = [int(t) for t in tokens]
    n = 0
    g = 0
    masked: List[int] = []
    seen_ngrams: set = set()

    for t in range(start, len(tokens)):
        ctx = tokens[max(0, t - context_width):t]

        if ngram_block and t >= ngram_block - 1:
            gram = tuple(tokens[t - ngram_block + 1:t + 1])
            if gram in seen_ngrams:
                masked.append(t)
                continue
        else:
            gram = None

        gl = green_list(
            key, ctx, context_width, gamma, vocab_size,
        )
        n += 1
        if tokens[t] in gl:
            g += 1

        if gram is not None:
            seen_ngrams.add(gram)

    return n, g, masked


def z_score(g: int, n: int, gamma: float) -> float:
    """标准化统计量 z = (g - γn) / √(nγ(1-γ))。"""
    if n <= 0:
        return 0.0
    mean = gamma * n
    var = n * gamma * (1.0 - gamma)
    if var <= 0:
        return 0.0
    return (g - mean) / math.sqrt(var)


def p_value(z: float) -> float:
    """单侧上尾 p 值 = 1 - Φ(z)，用 erf 精确计算。"""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# --------------------------------------------------------------------------- #
# 高层检测接口
# --------------------------------------------------------------------------- #
def detect(
    tokens: Sequence[int],
    key: bytes,
    gamma: float,
    context_width: int,
    vocab_size: int,
    ngram_block: int = 4,
    threshold: float | None = None,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    score_from: int | None = None,
    key_id: str = "",
    threshold_id: str = "",
) -> dict:
    """对一段 token 序列执行 KGW 检测，返回结构化结果。

    状态口径：
    - insufficient_length：有效位置 n < min_tokens，不进入有效判定；
    - completed：给出 z / p / n / g / 判定。
    """
    result = {
        "key_id": key_id,
        "threshold": threshold,
        "threshold_id": threshold_id,
        "min_tokens": min_tokens,
        "gamma": gamma,
        "context_width": context_width,
    }
    try:
        n, g, masked = count_green_hits(
            tokens, key, gamma, context_width, vocab_size,
            ngram_block=ngram_block, score_from=score_from,
        )
    except Exception as exc:  # noqa: BLE001 — 检测失败要显式记录，不能静默
        result.update(status=STATUS_FAILED, error=str(exc), n=0, g=0,
                      z=0.0, p=1.0, masked=[], decision=None)
        return result

    z = z_score(g, n, gamma)
    p = p_value(z)
    result.update(n=n, g=g, z=z, p=p, masked=masked)

    if n < min_tokens:
        result.update(status=STATUS_INSUFFICIENT_LENGTH, decision=None)
        return result

    decision = (z >= threshold) if threshold is not None else None
    result.update(status=STATUS_COMPLETED, decision=decision)
    return result


# --------------------------------------------------------------------------- #
# 生成端 logits 偏置（延迟导入 torch）
# --------------------------------------------------------------------------- #
class KGWLogitsProcessor:
    """Hugging Face ``LogitsProcessor`` 风格的 KGW 偏置器。

    在每一步解码时，依据前 ``context_width`` 个 token 重新划分绿名单，
    对绿名单 token 的 logits 加上 ``delta``。``delta=0`` 时不改变分布，
    作为真正的无水印基线。
    """

    def __init__(
        self,
        key: bytes,
        gamma: float,
        delta: float,
        context_width: int,
        vocab_size: int,
        ngram_block: int = 4,
        algo_version: str = "kgw-v1",
    ) -> None:
        self.key = key
        self.gamma = gamma
        self.delta = float(delta)
        self.context_width = int(context_width)
        self.vocab_size = int(vocab_size)
        self.ngram_block = int(ngram_block)
        self.algo_version = algo_version
        self._torch = None

    def _get_torch(self):
        if self._torch is None:
            import torch  # 延迟导入：无 torch 时仍可测试核心算法
            self._torch = torch
        return self._torch

    def __call__(self, input_ids, scores):
        torch = self._get_torch()
        if self.delta == 0.0:
            return scores
        try:
            ids = input_ids[0].tolist()
        except Exception:
            ids = list(input_ids)
        ctx = ids[-self.context_width:]
        gl = green_list(
            self.key, ctx, self.context_width, self.gamma,
            self.vocab_size, self.algo_version,
        )
        if gl:
            idx = torch.tensor(sorted(gl), dtype=torch.long, device=scores.device)
            scores[0, idx] = scores[0, idx] + self.delta
        return scores
