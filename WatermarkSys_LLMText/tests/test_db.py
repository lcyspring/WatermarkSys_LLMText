"""数据库层单测：建表、写入、计数。"""
from wm.config import WatermarkConfig
from wm.db import connect, counts, ensure_config, init_db, insert_detection, insert_generation


def test_init_and_counts(tmp_path):
    db = str(tmp_path / "t.sqlite")
    init_db(db)
    conn = connect(db)
    try:
        c = counts(conn)
        assert c["generations"] == 0
    finally:
        conn.close()


def test_insert_config_and_generation(tmp_path):
    db = str(tmp_path / "t.sqlite")
    init_db(db)
    conn = connect(db)
    try:
        cfg = WatermarkConfig(vocab_size=512, delta=2.0)
        ensure_config(conn, cfg)
        insert_generation(conn, {
            "gen_id": "g1", "prompt_id": "P1", "config_hash": cfg.config_hash(),
            "seed": 0, "delta": 2.0, "text": "hello", "n_new_tokens": 3,
            "elapsed_ms": 5, "status": "completed",
        })
        insert_detection(conn, {
            "det_id": "d1", "gen_id": "g1", "source": "generation",
            "config_hash": cfg.config_hash(), "threshold_id": "tau-v1",
            "n": 100, "g": 40, "z": 5.0, "p": 0.001, "decision": 1,
            "status": "completed",
        })
        conn.commit()
        c = counts(conn)
        assert c["watermark_configs"] == 1
        assert c["generations"] == 1
        assert c["detections"] == 1
        row = conn.execute("SELECT z, decision FROM detections").fetchone()
        assert row["z"] == 5.0 and row["decision"] == 1
    finally:
        conn.close()
