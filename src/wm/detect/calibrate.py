"""阈值校准：只由独立负例的 z 分数选择经验阈值。

口径（与任务书/开题报告一致）：
- 阈值 tau 仅由校准集负例确定，目标 FPR=1%；
- 三个 delta 档共用同一阈值；
- 400 负例的误报步长为 0.25 个百分点，零误报不能证明总体 FPR=0，
  因此同时给出二项分布置信区间。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def calibrate_threshold(neg_z: list[float], target_fpr: float = 0.01) -> float:
    """取负例 z 分数的 (1 - target_fpr) 分位数作为经验阈值。"""
    if not neg_z:
        raise ValueError("neg_z must not be empty")
    scores = sorted(float(z) for z in neg_z)
    # 上尾：阈值取使得约 target_fpr 比例的负例超过它
    k = int(math.ceil((1.0 - target_fpr) * len(scores)))
    k = min(max(k, 0), len(scores) - 1)
    return scores[k]


def fpr_binomial_ci(fp: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson 二项比例置信区间，用于报告 FPR 的不确定度。"""
    if n <= 0:
        return (0.0, 0.0)
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    p_hat = fp / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class CalibrationResult:
    threshold: float
    n_neg: int
    target_fpr: float
    emp_fpr: float
    fpr_ci: tuple[float, float]
    threshold_id: str
