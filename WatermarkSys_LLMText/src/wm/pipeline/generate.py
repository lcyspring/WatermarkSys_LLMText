"""生成流水线：提示 → 多档 delta 生成 → JSONL 落盘（可断点续跑）。

记录字段固定，保证 W7 批量生成、W8 校准、W12 主结果共用同一数据源。
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterable, Sequence

from wm.config import WatermarkConfig
from wm.detect.detector import Detector
from wm.model_adapter import ModelAdapter, SyntheticLM
from wm.watermark.kgw import KGWLogitsProcessor

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def _write_jsonl(records: Iterable[dict], path: str, append: bool = True) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_processor(config: WatermarkConfig, key: bytes) -> KGWLogitsProcessor:
    return KGWLogitsProcessor(
        key=key, gamma=config.gamma, delta=config.delta,
        context_width=config.context_width, vocab_size=config.vocab_size,
        ngram_block=config.ngram_block, algo_version=config.algo_version,
    )


def make_detector(config: WatermarkConfig, key: bytes,
                  threshold: float | None = None,
                  threshold_id: str = "") -> Detector:
    return Detector(config, key, threshold=threshold,
                    threshold_id=threshold_id)


class GeneratePipeline:
    """对一组提示按给定配置批量生成。"""

    def __init__(self, adapter, key: bytes, output_path: str) -> None:
        self.adapter = adapter     # SyntheticLM 或 ModelAdapter
        self.key = key
        self.output_path = output_path

    def _generate_one(self, prompt: str, prompt_id: str,
                      config: WatermarkConfig) -> dict:
        t0 = time.time()
        prompt_ids = self.adapter.encode(prompt)
        if isinstance(self.adapter, ModelAdapter):
            proc = make_processor(config, self.key) if config.delta != 0 else None
            new_ids = self.adapter.generate_ids(
                prompt_ids, logits_processor=proc,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature, top_p=config.top_p,
                seed=config.seed,
            )
            text = self.adapter.decode(new_ids)
            mode = "hf"
        else:
            new_ids = self.adapter.generate_ids(
                prompt_ids, config, self.key,
                max_new_tokens=config.max_new_tokens)
            text = self.adapter.decode(new_ids)
            mode = "synthetic"

        info = self.adapter.info() if hasattr(self.adapter, "info") else None
        return {
            "gen_id": f"{config.config_hash()}-{prompt_id}",
            "prompt_id": prompt_id,
            "prompt": prompt,
            "prompt_len": len(prompt_ids),
            "prompt_token_ids": [int(t) for t in prompt_ids],
            "config_hash": config.config_hash(),
            "key_id": config.key_id,
            "delta": config.delta,
            "gamma": config.gamma,
            "context_width": config.context_width,
            "mode": mode,
            "model_id": config.model_id or (info.model_id if info else ""),
            "n_new_tokens": len(new_ids),
            "token_ids": [int(t) for t in new_ids],
            "text": text,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "status": STATUS_COMPLETED,
        }

    def run(self, prompts: Iterable[tuple[str, str]],
            config: WatermarkConfig, append: bool = True) -> dict:
        """prompts 为 (prompt_id, prompt_text) 序列；返回统计摘要。"""
        ok, fail = 0, 0
        records = []
        for prompt_id, prompt in prompts:
            try:
                records.append(self._generate_one(prompt, prompt_id, config))
                ok += 1
            except Exception as exc:  # noqa: BLE001 — 失败必须记录，可重跑
                fail += 1
                records.append({
                    "prompt_id": prompt_id, "prompt": prompt,
                    "config_hash": config.config_hash(),
                    "key_id": config.key_id, "delta": config.delta,
                    "status": STATUS_FAILED, "error": str(exc),
                })
        _write_jsonl(records, self.output_path, append=append)
        return {"ok": ok, "fail": fail, "path": self.output_path}


def build_adapter(mode: str = "synthetic", vocab_size: int = 8192,
                  model_id: str = "", revision: str | None = None,
                  dtype: str = "fp32", device: str | None = None):
    """按 mode 构造适配器；真实模型需已安装 transformers 且权重可获取。"""
    if mode == "synthetic":
        return SyntheticLM(vocab_size=vocab_size)
    if mode == "hf":
        return ModelAdapter(model_id=model_id, revision=revision,
                            dtype=dtype, device=device)
    raise ValueError(f"unknown mode: {mode}")


def load_prompts(path: str) -> list[tuple[str, str]]:
    return [(r["prompt_id"], r["text"]) for r in read_jsonl(path)]


def detect_record(record: dict, config: WatermarkConfig, key: bytes,
                  threshold: float | None, threshold_id: str = "") -> dict:
    """对单条生成记录做检测（token 级，排除提示前缀）。"""
    from wm.watermark import detect as kgw_detect

    ids = list(record.get("prompt_token_ids", [])) + list(record["token_ids"])
    score_from = record.get("prompt_len")
    res = kgw_detect(
        ids, key=key, gamma=config.gamma,
        context_width=config.context_width, vocab_size=config.vocab_size,
        ngram_block=config.ngram_block, threshold=threshold,
        min_tokens=32, score_from=score_from,
        key_id=config.key_id, threshold_id=threshold_id,
    )
    return {
        "gen_id": record.get("gen_id"),
        "prompt_id": record.get("prompt_id"),
        "delta": record.get("delta"),
        "mode": record.get("mode"),
        "status": res["status"],
        "n": res["n"], "g": res["g"], "z": res["z"], "p": res["p"],
        "decision": res["decision"],
        "masked": len(res.get("masked", [])),
    }


def ensure_sequence(record: dict) -> Sequence[int]:
    return list(record.get("token_ids", []))
