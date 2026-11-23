"""水印算法包。"""
from wm.watermark.kgw import (
    KGWLogitsProcessor,
    STATUS_COMPLETED,
    STATUS_CONFIG_MISMATCH,
    STATUS_FAILED,
    STATUS_INSUFFICIENT_LENGTH,
    STATUS_PLANNED,
    count_green_hits,
    detect,
    green_list,
    p_value,
    z_score,
)

__all__ = [
    "KGWLogitsProcessor",
    "green_list",
    "count_green_hits",
    "z_score",
    "p_value",
    "detect",
    "STATUS_PLANNED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_INSUFFICIENT_LENGTH",
    "STATUS_CONFIG_MISMATCH",
]
