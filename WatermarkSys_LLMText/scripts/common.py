"""脚本共享工具：路径、密钥、配置与 JSONL 读写。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from wm.config import WatermarkConfig  # noqa: E402
from wm.pipeline.generate import read_jsonl as _read  # noqa: E402

DOCS_ROOT = CODE_ROOT.parent / "docs"
OUT = CODE_ROOT / "outputs"
REPORTS = DOCS_ROOT / "reports"
DATA = CODE_ROOT / "data"
PROMPTS = DATA / "prompts" / "prompts_v1.jsonl"

DEFAULT_KEY_HEX = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
WRONG_KEY_HEX = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

GEN_PATH = OUT / "generations.jsonl"
EDIT_PATH = OUT / "edits.jsonl"
DET_PATH = OUT / "detections.jsonl"
THRESHOLD_PATH = OUT / "threshold_v1.json"
METRICS_PATH = OUT / "metrics_summary.json"
DB_PATH = OUT / "wm.sqlite"


def ensure_dirs() -> None:
    for p in (OUT, REPORTS, DATA / "prompts"):
        p.mkdir(parents=True, exist_ok=True)


def load_key(key_id: str = "default", wrong: bool = False) -> bytes:
    """优先读取 secrets.local.yaml；缺失时使用公开的开发用占位密钥。"""
    if wrong:
        return bytes.fromhex(WRONG_KEY_HEX)
    secrets = CODE_ROOT / "configs" / "secrets.local.yaml"
    if secrets.exists():
        try:
            from wm.config import load_yaml
            data = load_yaml(str(secrets))
            hexval = data.get(f"keys.{key_id}.hex") or data.get(f"{key_id}.hex")
            if hexval:
                return bytes.fromhex(str(hexval))
        except Exception:
            pass
    return bytes.fromhex(DEFAULT_KEY_HEX)


def get_config(delta: float = 0.0, vocab_size: int = 2048,
               mode: str = "synthetic", seed: int = 0,
               max_new_tokens: int = 200,
               model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
               temperature: float = 1.0) -> WatermarkConfig:
    return WatermarkConfig(
        delta=float(delta),
        vocab_size=int(vocab_size),
        model_id="" if mode == "synthetic" else model_id,
        device="cpu",
        dtype="fp32",
        seed=int(seed),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )


def read_jsonl(path) -> list[dict]:
    return _read(str(path))


def load_prompts(path=None) -> list[tuple[str, str]]:
    """读取提示集，返回 (prompt_id, text) 列表。"""
    return [(r["prompt_id"], r["text"]) for r in read_jsonl(path or PROMPTS)]


def write_jsonl(records: list[dict], path, append: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj, path) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_threshold() -> tuple[float, str]:
    if not THRESHOLD_PATH.exists():
        raise SystemExit("缺少阈值文件，请先运行 scripts/05_calibrate.py")
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["threshold"]), str(d.get("threshold_id", "tau-v1"))


def markdown_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
