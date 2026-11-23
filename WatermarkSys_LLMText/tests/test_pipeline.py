"""流水线端到端单测（离线机制模拟，不需要模型权重）。"""
from wm.config import WatermarkConfig
from wm.model_adapter import SyntheticLM
from wm.pipeline.generate import GeneratePipeline, detect_record

V = 512
KEY = b"unit-test-key"


def _cfg(delta: float, seed: int = 0) -> WatermarkConfig:
    return WatermarkConfig(delta=delta, vocab_size=V, max_new_tokens=120,
                           seed=seed)


def _run(delta: float, prompts, tmp_path, name: str):
    lm = SyntheticLM(vocab_size=V)
    path = tmp_path / f"{name}.jsonl"
    GeneratePipeline(lm, KEY, str(path)).run(prompts, _cfg(delta))
    import json
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def test_generation_record_fields(tmp_path):
    recs = _run(0.0, [("P1", "hello world test")], tmp_path, "g0")
    r = recs[0]
    for k in ("gen_id", "prompt_id", "config_hash", "token_ids", "text",
              "status", "delta", "mode", "prompt_len"):
        assert k in r
    assert r["status"] == "completed"
    assert len(r["token_ids"]) == 120
    assert r["mode"] == "synthetic"


def test_delta_zero_z_near_zero(tmp_path):
    prompts = [(f"P{i}", f"prompt number {i} about watermarks") for i in range(12)]
    recs = _run(0.0, prompts, tmp_path, "g0")
    zs = []
    for r in recs:
        d = detect_record(r, _cfg(0.0), KEY, threshold=None)
        if d["status"] == "completed":
            zs.append(d["z"])
    assert zs
    assert abs(sum(zs) / len(zs)) < 1.5


def test_delta_two_detected(tmp_path):
    prompts = [(f"P{i}", f"prompt number {i} about watermarks") for i in range(12)]
    recs = _run(2.0, prompts, tmp_path, "g2")
    hits = 0
    for r in recs:
        d = detect_record(r, _cfg(2.0), KEY, threshold=2.5)
        if d["status"] == "completed" and d["decision"]:
            hits += 1
    assert hits >= len(recs) - 1  # 高偏置下几乎全部检出


def test_delta_monotone(tmp_path):
    prompts = [(f"P{i}", f"topic {i}") for i in range(8)]
    means = []
    for d in (0.0, 1.0, 2.0):
        recs = _run(d, prompts, tmp_path, f"g{int(d*10)}")
        zs = [detect_record(r, _cfg(d), KEY, None)["z"] for r in recs]
        means.append(sum(zs) / len(zs))
    assert means[0] < means[1] < means[2]


def test_prompt_prefix_excluded_from_scoring(tmp_path):
    recs = _run(2.0, [("P1", "a b c d e f g h")], tmp_path, "g2")
    r = recs[0]
    d = detect_record(r, _cfg(2.0), KEY, threshold=None)
    assert d["n"] <= len(r["token_ids"])
