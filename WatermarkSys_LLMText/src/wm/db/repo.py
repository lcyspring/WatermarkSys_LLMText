"""SQLite 数据访问层：建库、写入、查询，保证数据血缘可追溯。"""
from __future__ import annotations

import json
import os
import sqlite3

from wm.db.models import SCHEMA


def connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str) -> None:
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def ensure_config(conn: sqlite3.Connection, config) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO watermark_configs(config_hash, config_json, key_id)"
        " VALUES (?,?,?)",
        (config.config_hash(),
         json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True),
         config.key_id),
    )


def insert_generation(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO generations("
        "gen_id,prompt_id,config_hash,seed,delta,text,new_tokens,elapsed_ms,"
        "status,error) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (rec.get("gen_id"), rec.get("prompt_id"), rec.get("config_hash"),
         rec.get("seed"), rec.get("delta"), rec.get("text"),
         rec.get("n_new_tokens"), rec.get("elapsed_ms"),
         rec.get("status"), rec.get("error")),
    )


def insert_detection(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO detections("
        "det_id,gen_id,source,config_hash,threshold_id,n,g,z,p,decision,status)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (rec.get("det_id"), rec.get("gen_id"), rec.get("source"),
         rec.get("config_hash"), rec.get("threshold_id"),
         rec.get("n"), rec.get("g"), rec.get("z"), rec.get("p"),
         rec.get("decision"), rec.get("status")),
    )


def insert_edit(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO edits("
        "edit_id,gen_id,type,target_rate,actual_rate,length_before,"
        "length_after,text,status,error) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (rec.get("edit_id"), rec.get("gen_id"), rec.get("type"),
         rec.get("target_rate"), rec.get("actual_rate"),
         rec.get("length_before"), rec.get("length_after"),
         rec.get("text"), rec.get("status"), rec.get("error")),
    )


def insert_metrics(conn: sqlite3.Connection, scope: str, label: str,
                   tp: int, fp: int, tn: int, fn: int,
                   fpr: float, fnr: float, payload: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO metrics(scope,label,tp,fp,tn,fn,fpr,fnr,payload)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (scope, label, tp, fp, tn, fn, fpr, fnr,
         json.dumps(payload or {}, ensure_ascii=False)),
    )


def counts(conn: sqlite3.Connection) -> dict:
    out = {}
    for table in ("watermark_configs", "prompts", "generations",
                  "detections", "edits", "metrics"):
        out[table] = conn.execute(
            f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
    return out
