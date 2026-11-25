"""KGW 核心算法单测（纯 Python，无 torch 也可运行）。

运行：python -m pytest tests -q   或   python tests/test_kgw.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math  # noqa: E402

from wm.config import WatermarkConfig  # noqa: E402
from wm.watermark import kgw  # noqa: E402
from wm.watermark.kgw import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_INSUFFICIENT_LENGTH,
    count_green_hits,
    detect,
    green_list,
    p_value,
    z_score,
)

KEY_A = b"test-key-A"
KEY_B = b"test-key-B"
V = 1000
GAMMA = 0.25
H = 4


def test_config_hash_changes_with_delta():
    c0 = WatermarkConfig(delta=0.0)
    c1 = WatermarkConfig(delta=2.0)
    assert c0.config_hash() != c1.config_hash()
    assert c0.config_hash() == WatermarkConfig(delta=0.0).config_hash()


def test_green_list_deterministic():
    ctx = [1, 2, 3, 4]
    g1 = green_list(KEY_A, ctx, H, GAMMA, V)
    g2 = green_list(KEY_A, ctx, H, GAMMA, V)
    assert g1 == g2
    # 不同密钥 → 划分不同（极大概率）
    g3 = green_list(KEY_B, ctx, H, GAMMA, V)
    assert g1 != g3


def test_green_list_size_matches_gamma():
    g = green_list(KEY_A, list(range(50, 54)), H, GAMMA, V)
    expected = max(1, round(GAMMA * V))
    assert len(g) == expected


def test_green_list_depends_on_context():
    g1 = green_list(KEY_A, [1, 2, 3, 4], H, GAMMA, V)
    g2 = green_list(KEY_A, [1, 2, 3, 5], H, GAMMA, V)
    assert g1 != g2


def test_z_score_baseline_and_extreme():
    n = 400
    g = round(GAMMA * n)
    assert abs(z_score(g, n, GAMMA)) < 0.5  # 命中期望 → z 接近 0
    assert z_score(n, n, GAMMA) > 0          # 全绿 → z 显著为正
    assert z_score(0, n, GAMMA) < 0          # 全红 → z 显著为负


def test_p_value_symmetry():
    assert abs(p_value(0.0) - 0.5) < 1e-9
    assert p_value(3.0) < 0.01
    assert p_value(-3.0) > 0.99


def test_detect_insufficient_length():
    tokens = list(range(20))  # 太短，n < 32
    res = detect(tokens, KEY_A, GAMMA, H, V, min_tokens=32)
    assert res["status"] == STATUS_INSUFFICIENT_LENGTH
    assert res["decision"] is None


def _build_all_green(length=80):
    """构造一段每个生成 token 都命中其位置绿名单的序列。"""
    seq = list(range(H))  # 前缀上下文
    for _ in range(length - H):
        ctx = seq[-H:]
        gl = green_list(KEY_A, ctx, H, GAMMA, V)
        seq.append(min(gl))  # 选绿名单中最小 id
    return seq


def test_detect_all_green_high_z():
    seq = _build_all_green(80)
    res = detect(seq, KEY_A, GAMMA, H, V, threshold=3.0, min_tokens=32)
    assert res["status"] == STATUS_COMPLETED
    assert res["n"] >= 32
    assert res["g"] == res["n"]  # 全命中
    assert res["z"] > 10
    assert res["decision"] is True


def test_detect_prompt_prefix_excluded():
    """检测只统计 score_from 之后的位置，前缀不计入 n（屏蔽位置也不计入）。"""
    seq = _build_all_green(80)
    n_full, _, masked_full = count_green_hits(seq, KEY_A, GAMMA, H, V)
    n_shift, _, masked_shift = count_green_hits(
        seq, KEY_A, GAMMA, H, V, score_from=H + 5)
    assert n_full + len(masked_full) == len(seq) - H
    assert n_shift + len(masked_shift) == len(seq) - (H + 5)


def test_repetition_masking():
    """重复 n-gram 位置被屏蔽，n 减少。"""
    # 构造一段明显重复的文本（如 "10 11 12 13" 反复出现）
    unit = [10, 11, 12, 13]
    seq = list(range(H)) + unit * 6
    n_nomask, _, _ = count_green_hits(seq, KEY_A, GAMMA, H, V, ngram_block=0)
    n_mask, _, masked = count_green_hits(seq, KEY_A, GAMMA, H, V, ngram_block=4)
    assert len(masked) > 0
    assert n_mask < n_nomask


def test_special_tokens_excluded():
    """特殊 token（EOS/PAD）不计入 n、g 与重复 n-gram，结果与无特殊 token 一致。"""
    seq = _build_all_green(80)
    base = detect(seq, KEY_A, GAMMA, H, V, threshold=3.0, min_tokens=32)
    special = 999  # 正常序列中不会出现的特殊 token
    res = detect(seq + [special, special, special], KEY_A, GAMMA, H, V,
                 threshold=3.0, min_tokens=32, special_ids=[special])
    assert res["n"] == base["n"]
    assert res["g"] == base["g"]
    assert abs(res["z"] - base["z"]) < 1e-12
    assert len(res["masked"]) == len(base["masked"]) + 3


def test_detect_negative_control_uniform_random():
    """无水印文本（均匀随机 token）的 z 不应系统性偏高。"""
    import random
    rng = random.Random(42)
    zs = []
    for _ in range(20):
        seq = [rng.randrange(V) for _ in range(120)]
        res = detect(seq, KEY_A, GAMMA, H, V, min_tokens=32)
        zs.append(res["z"])
    mean_z = sum(zs) / len(zs)
    assert abs(mean_z) < 1.0  # 随机文本平均 z 接近 0，而非显著正


def test_delta_zero_is_true_baseline():
    """delta=0 时偏置器不修改 scores（无 torch 时自动跳过）。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("torch not installed")
    proc = kgw.KGWLogitsProcessor(KEY_A, GAMMA, 0.0, H, V)
    import torch
    scores = torch.randn(1, V)
    out = proc(torch.tensor([[1, 2, 3, 4]]), scores.clone())
    assert torch.equal(out, scores)


if __name__ == "__main__":
    sys.exit(0 if __import__("pytest").main([__file__, "-q"]) else 1)
