"""token 级后处理与质量代理指标单测。"""
from wm.eval.quality import (
    compression_ratio,
    distinct_n,
    quality_report,
    repetition_rate,
)
from wm.postprocess.token_edits import (
    EDIT_LEVELS,
    apply_edit,
    delete_random,
    substitute_random,
    truncate,
)


def test_delete_random_keeps_order_and_rate():
    toks = list(range(100))
    r = delete_random(toks, 0.2, seed=3)
    assert r.original_count == 100
    assert r.edited_count == 80
    assert abs(r.actual_rate - 0.2) < 1e-9
    assert r.tokens == sorted(r.tokens)  # 保序


def test_truncate_keeps_prefix():
    toks = list(range(40))
    r = truncate(toks, 0.5)
    assert r.tokens == list(range(20))
    assert r.actual_rate == 0.5


def test_substitute_changes_values_not_length():
    toks = [1, 2, 3, 4, 5, 6, 7, 8]
    r = substitute_random(toks, 0.5, vocab_size=1000, seed=7)
    assert len(r.tokens) == len(toks)
    changed = sum(1 for a, b in zip(toks, r.tokens) if a != b)
    assert changed == 4


def test_apply_edit_dispatch():
    toks = list(range(50))
    for kind in EDIT_LEVELS:
        r = apply_edit(toks, kind, EDIT_LEVELS[kind][0], vocab_size=1000)
        assert r.status == "completed"
        assert 0 < r.edited_count <= len(toks)


def test_apply_edit_unknown_kind():
    try:
        apply_edit([1, 2], "nope", 0.1, 100)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("应抛出 ValueError")


def test_quality_metrics_range():
    text = "the quick brown fox jumps over the lazy dog again and again"
    assert 0.0 <= distinct_n(text, 1) <= 1.0
    assert 0.0 <= distinct_n(text, 2) <= 1.0
    assert 0.0 <= repetition_rate(text, 4) <= 1.0
    assert compression_ratio(text) > 0.0


def test_quality_report_aggregates():
    q = quality_report(["a b c d e", "a a a a a"])
    assert q["count"] == 2
    assert "len_mean" in q and "distinct2_mean" in q
    assert q["repeat4_mean"] >= 0.0
