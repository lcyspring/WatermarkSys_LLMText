"""阈值校准扩展单测：目标 FPR 不被四舍五入突破、阈值可持久化。"""
import json

from wm.detect.calibrate import calibrate, calibrate_threshold, fpr_binomial_ci


def test_calibrate_respects_target_fpr():
    neg = [((i * 37) % 101) / 10.0 for i in range(400)]
    res = calibrate(neg, target_fpr=0.01)
    fp = sum(1 for z in neg if z >= res.threshold)
    assert fp / len(neg) <= 0.0101
    assert res.emp_fpr <= 0.0101


def test_calibrate_threshold_monotone_in_fpr():
    neg = sorted(float(i) for i in range(100))
    t1 = calibrate_threshold(neg, 0.01)
    t5 = calibrate_threshold(neg, 0.05)
    assert t1 >= t5


def test_threshold_deterministic():
    neg = [0.1 * i for i in range(50)]
    a = calibrate(neg)
    b = calibrate(neg)
    assert a.threshold == b.threshold
    assert a.emp_fpr == b.emp_fpr


def test_ci_zero_fp_has_positive_upper():
    lo, hi = fpr_binomial_ci(0, 400)
    assert lo == 0.0
    assert 0.0 < hi < 0.02


def test_save_and_load(tmp_path):
    from wm.detect.calibrate import load_threshold, save_threshold
    res = calibrate([0.1, 0.2, 0.3, 0.4], target_fpr=0.25)
    p = tmp_path / "tau.json"
    save_threshold(res, str(p))
    d = load_threshold(str(p))
    assert json.loads(json.dumps(d))["threshold"] == res.threshold
    assert d["threshold_id"].startswith("tau-")
