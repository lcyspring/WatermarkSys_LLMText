"""评估指标扩展单测：AUC、TPR@FPR、长度分层、Bootstrap。"""
from wm.eval.metrics import (
    bootstrap_ci,
    confusion_at_threshold,
    length_bin,
    roc_auc,
    stratified_by_length,
    summarize_z,
    tpr_at_fpr,
)


def test_roc_auc_perfect_and_chance():
    assert roc_auc([5, 6, 7, 1, 2, 3], [True, True, True, False, False, False]) == 1.0
    assert roc_auc([3, 3, 3, 3], [True, False, True, False]) == 0.5


def test_tpr_at_fpr_threshold():
    neg = [-1.0, -0.5, 0.0, 0.5, 1.0]
    pos = [3.0, 4.0, 5.0]
    zs = neg + pos
    labels = [False] * 5 + [True] * 3
    tau, tpr = tpr_at_fpr(zs, labels, 0.2)
    assert tpr == 1.0
    assert tau <= 3.0


def test_confusion_at_threshold():
    zs = [0.1, 0.2, 5.0, 6.0]
    labels = [False, False, True, True]
    m = confusion_at_threshold(zs, labels, 3.0)
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 0, 2, 0)
    assert m.accuracy == 1.0


def test_length_bin_boundaries():
    assert length_bin(31) == "32-63"
    assert length_bin(32) == "32-63"
    assert length_bin(64) == "64-127"
    assert length_bin(128) == "128-255"
    assert length_bin(500) == "256+"


def test_stratified_by_length():
    recs = [{"n": 40, "z": 1}, {"n": 70, "z": 2}, {"n": 300, "z": 3}]
    out = stratified_by_length(recs)
    assert list(out.keys()) == ["32-63", "64-127", "256+"]


def test_bootstrap_ci_contains_mean():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    mean, lo, hi = bootstrap_ci(vals, n_boot=200, seed=1)
    assert lo <= mean <= hi
    assert abs(mean - 3.0) < 1e-9


def test_summarize_z_ignores_failed():
    recs = [{"z": 1.0, "status": "completed"},
            {"z": 2.0, "status": "completed"},
            {"z": 99.0, "status": "insufficient_length"}]
    s = summarize_z(recs)
    assert s["count"] == 2
    assert s["max_z"] <= 2.0
