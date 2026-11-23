"""SQLite 数据模型：建表语句集中管理，保证数据血缘可追溯。"""
from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS watermark_configs (
    config_hash TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    key_id TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id TEXT PRIMARY KEY,
    source_doc TEXT,
    split TEXT,            -- dev / calib / test
    task_type TEXT,
    text TEXT NOT NULL,
    text_hash TEXT,
    license TEXT,
    revision TEXT
);

CREATE TABLE IF NOT EXISTS generations (
    gen_id TEXT PRIMARY KEY,
    prompt_id TEXT,
    config_hash TEXT,
    seed INTEGER,
    delta REAL,
    text TEXT,
    new_tokens INTEGER,
    elapsed_ms INTEGER,
    status TEXT,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (config_hash) REFERENCES watermark_configs(config_hash)
);

CREATE TABLE IF NOT EXISTS detections (
    det_id TEXT PRIMARY KEY,
    gen_id TEXT,
    source TEXT,             -- generation / natural / edited
    config_hash TEXT,
    threshold_id TEXT,
    n INTEGER, g INTEGER, z REAL, p REAL,
    decision INTEGER,
    status TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edits (
    edit_id TEXT PRIMARY KEY,
    gen_id TEXT,
    type TEXT,               -- delete / truncate / synonym
    target_rate REAL,
    actual_rate REAL,
    length_before INTEGER,
    length_after INTEGER,
    text TEXT,
    status TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT,              -- delta / edit_type / length_bin
    label TEXT,
    tp INTEGER, fp INTEGER, tn INTEGER, fn INTEGER,
    fpr REAL, fnr REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT,
    detail TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""
