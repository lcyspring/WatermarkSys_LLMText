"""评估指标：混淆矩阵及 Accuracy/Precision/Recall/F1/FPR/FNR。

纯 Python 实现，便于在无额外依赖环境下验证计算口径。
正类约定为“含本方案水印的文本”。
"""
from __future__ import annotations

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
            "accuracy": self.accuracy, "precision": self.precision,
            "recall": self.recall, "f1": self.f1,
            "fpr": self.fpr, "fnr": self.fnr,
        }


def confusion(y_true: list[bool], y_pred: list[bool]) -> BinaryMetrics:
    """由真实标签与预测标签计算混淆矩阵。"""
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
