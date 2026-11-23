import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wm.detect import calibrate_threshold, fpr_binomial_ci  # noqa: E402
from wm.eval import confusion  # noqa: E402
from wm.postprocess import delete_words, replace_synonyms, truncate  # noqa: E402


def test_calibrate_threshold_quantile():
    neg = list(range(100))  # 0..99
    tau = calibrate_threshold(neg, target_fpr=0.01)
    assert tau == 99  # 1% 上尾 → 取最大
    tau5 = calibrate_threshold(neg, target_fpr=0.05)
    assert tau5 == 95


def test_fpr_ci_bounds():
    lo, hi = fpr_binomial_ci(0, 400)
    assert lo == 0.0 and hi < 0.02  # 零误报的区间上限仍 > 0
    lo2, hi2 = fpr_binomial_ci(4, 400)
    assert lo2 <= 0.01 <= hi2


def test_confusion_metrics():
    y_true = [True, True, False, False, True]
    y_pred = [True, False, True, False, True]
    m = confusion(y_true, y_pred)
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1)
    assert abs(m.recall - 2 / 3) < 1e-9
    assert abs(m.fpr - 0.5) < 1e-9
    d = m.to_dict()
    assert d["n_pos"] == 3 and d["n_neg"] == 2


def test_delete_words_actual_rate():
    text = "a b c d e f g h i j"
    r = delete_words(text, 0.3, seed=0)
    assert r.original_count == 10
    assert r.edited_count == 7
    assert abs(r.actual_rate - 0.3) < 1e-9


def test_truncate_keeps_prefix():
    text = "a b c d e f g h"
    r = truncate(text, 0.5)
    assert r.kept_count == 4
    assert r.text.split() == ["a", "b", "c", "d"]


def test_synonym_replaces_some():
    text = "good big small happy said bad"
    r = replace_synonyms(text, 0.5, seed=0)
    assert r.candidate_count > 0
    assert r.replaced_count > 0
    assert r.replaced_count <= r.candidate_count
