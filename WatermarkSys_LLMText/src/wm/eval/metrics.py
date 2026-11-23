"""评估指标：混淆矩阵、阈值指标、ROC/AUC、长度分层、Bootstrap 置信区间。

纯 Python + 标准库实现，避免评估环节引入额外依赖。
正类约定为“含本方案水印的文本”。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def n_pos(self) -> int:
        return self.tp + self.fn

    @property
    def n_neg(self) -> int:
        return self.tn + self.fp

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def fnr(self) -> float:
        d = self.tp + self.fn
        return self.fn / d if d else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "n_pos": self.n_pos, "n_neg": self.n_neg,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
        }


def confusion(y_true: list[bool], y_pred: list[bool]) -> BinaryMetrics:
    m = BinaryMetrics()
    for t, p in zip(y_true, y_pred):
        if t and p:
            m.tp += 1
        elif (not t) and p:
            m.fp += 1
        elif (not t) and (not p):
            m.tn += 1
        else:
            m.fn += 1
    return m


def confusion_at_threshold(zs: list[float], labels: list[bool],
                           threshold: float) -> BinaryMetrics:
    return confusion(labels, [z >= threshold for z in zs])


def roc_auc(zs: list[float], labels: list[bool]) -> float:
    """AUC，等价于正负样本对的胜率（Mann-Whitney U 统计量）。"""
    pos = [z for z, y in zip(zs, labels) if y]
    neg = [z for z, y in zip(zs, labels) if not y]
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def tpr_at_fpr(zs: list[float], labels: list[bool], target_fpr: float) -> tuple[float, float]:
    """返回 (阈值, TPR)：阈值取使负例 FPR ≤ target_fpr 的最大值。"""
    neg = sorted([z for z, y in zip(zs, labels) if not y], reverse=True)
    if not neg:
        return (float("inf"), 0.0)
    k = int(math.floor(target_fpr * len(neg)))
    threshold = neg[k] if k < len(neg) else neg[-1]
    pos = [z for z, y in zip(zs, labels) if y]
    if not pos:
        return (threshold, 0.0)
    tpr = sum(1 for z in pos if z >= threshold) / len(pos)
    return (threshold, tpr)


LENGTH_BINS = ("32-63", "64-127", "128-255", "256+")


def length_bin(n: int) -> str:
    if n < 64:
        return "32-63"
    if n < 128:
        return "64-127"
    if n < 256:
        return "128-255"
    return "256+"


def stratified_by_length(records: list[dict]) -> dict[str, list[dict]]:
    """records 需含 n；按有效长度分箱，按区间由短到长返回。"""
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(length_bin(int(r["n"])), []).append(r)
    return {b: out[b] for b in LENGTH_BINS if b in out}


def bootstrap_ci(values: list[float], stat=None, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap 均值置信区间，返回 (mean, lower, upper)。"""
    if not values:
        return (0.0, 0.0, 0.0)
    stat = stat or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (stat(values), lo, hi)


def summarize_z(records: list[dict]) -> dict:
    """一组检测记录的 z 分布摘要。"""
    zs = [float(r["z"]) for r in records if r.get("status") == "completed"]
    if not zs:
        return {"count": 0}
    zs_sorted = sorted(zs)
    mean, lo, hi = bootstrap_ci(zs)
    return {
        "count": len(zs),
        "mean_z": round(mean, 3),
        "mean_z_ci95": [round(lo, 3), round(hi, 3)],
        "median_z": round(zs_sorted[len(zs_sorted) // 2], 3),
        "p05": round(zs_sorted[int(0.05 * len(zs_sorted))], 3),
        "p95": round(zs_sorted[min(int(0.95 * len(zs_sorted)),
                                  len(zs_sorted) - 1)], 3),
        "max_z": round(zs_sorted[-1], 3),
        "min_z": round(zs_sorted[0], 3),
    }
