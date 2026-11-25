"""检测器封装：持有配置与密钥，统一输出结构化检测结果。"""
from __future__ import annotations

from typing import Sequence

from wm.config import WatermarkConfig
from wm.watermark import kgw


class Detector:
    """KGW 检测器。生成与检测必须使用相同的 key/tokenizer/算法配置。"""

    def __init__(
        self,
        config: WatermarkConfig,
        key: bytes,
        threshold: float | None = None,
        threshold_id: str = "",
        min_tokens: int = kgw.DEFAULT_MIN_TOKENS,
        special_ids: Sequence[int] | None = None,
    ) -> None:
        self.config = config
        self.key = key
        self.threshold = threshold
        self.threshold_id = threshold_id
        self.min_tokens = min_tokens
        self.special_ids = list(special_ids) if special_ids else []

    def detect_tokens(
        self,
        tokens: Sequence[int],
        score_from: int | None = None,
    ) -> dict:
        return kgw.detect(
            tokens,
            key=self.key,
            gamma=self.config.gamma,
            context_width=self.config.context_width,
            vocab_size=self.config.vocab_size,
            ngram_block=self.config.ngram_block,
            threshold=self.threshold,
            min_tokens=self.min_tokens,
            score_from=score_from,
            special_ids=self.special_ids,
            key_id=self.config.key_id,
            threshold_id=self.threshold_id,
        )
