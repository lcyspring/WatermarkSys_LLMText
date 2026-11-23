"""配置一致性与 Web 接口单测。"""
import pytest

from wm.config import WatermarkConfig


def test_hash_changes_with_key_fields():
    base = WatermarkConfig(vocab_size=512, delta=2.0)
    variants = [
        WatermarkConfig(vocab_size=512, delta=0.0),
        WatermarkConfig(vocab_size=512, delta=2.0, gamma=0.30),
        WatermarkConfig(vocab_size=512, delta=2.0, context_width=5),
        WatermarkConfig(vocab_size=512, delta=2.0, algo_version="kgw-v2"),
        WatermarkConfig(vocab_size=512, delta=2.0, key_id="other"),
    ]
    hashes = {base.config_hash()} | {v.config_hash() for v in variants}
    assert len(hashes) == len(variants) + 1


def test_fingerprint_ignores_seed_but_hash_does_not():
    a = WatermarkConfig(vocab_size=512, seed=0)
    b = WatermarkConfig(vocab_size=512, seed=99)
    assert a.fingerprint() == b.fingerprint()
    assert a.config_hash() != b.config_hash()


def test_from_dict_ignores_unknown():
    cfg = WatermarkConfig.from_dict({"delta": 1.0, "unknown_field": 1})
    assert cfg.delta == 1.0
    assert not hasattr(cfg, "unknown_field")


def test_web_generate_and_detect():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from wm.web.app import app

    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.post("/api/generate", json={
        "prompt": "explain watermarking briefly", "delta": 2.0,
        "max_new_tokens": 64, "vocab_size": 512})
    assert r.status_code == 200
    body = r.json()
    assert body["n_new_tokens"] == 64
    assert body["detection"]["status"] in ("completed", "insufficient_length")
    assert client.get("/").status_code == 200
