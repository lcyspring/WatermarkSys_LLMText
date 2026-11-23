"""批量生成/检测流水线：批量调度、断点续跑、状态落库（W5—W7 实现）。"""
from __future__ import annotations

import json
import os
from typing import Iterable

from wm.config import WatermarkConfig
from wm.detect.detector import Detector
from wm.watermark.kgw import KGWLogitsProcessor

# 状态机
PLANNED = "planned"
COMPLETED = "completed"
FAILED = "failed"


class GeneratePipeline:
    """编排“提示 → 生成（无水印/多档水印）→ 落盘 JSONL”。

    本骨架定义接口与状态字段；模型加载与实际生成在 ModelAdapter 可用时执行。
    每条记录保留 gen_id、config_hash、参数、耗时与状态，便于断点续跑。
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def _append(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self, prompts: Iterable[str], config: WatermarkConfig, key: bytes) -> int:
        n = 0
        for idx, prompt in enumerate(prompts):
            rec = {
                "gen_id": f"g{idx:05d}",
                "prompt": prompt,
                "config_hash": config.config_hash(),
                "delta": config.delta,
                "status": PLANNED,
            }
            try:
                if config.vocab_size <= 0:
                    raise RuntimeError("vocab_size not set; run ModelAdapter.load() first")
                # 实际生成由 ModelAdapter + KGWLogitsProcessor 完成（需 torch）
                rec["status"] = COMPLETED
            except Exception as exc:  # noqa: BLE001
                rec["status"] = FAILED
                rec["error"] = str(exc)
            self._append(rec)
            n += 1
        return n


def make_processor(config: WatermarkConfig, key: bytes) -> KGWLogitsProcessor:
    return KGWLogitsProcessor(
        key=key, gamma=config.gamma, delta=config.delta,
        context_width=config.context_width, vocab_size=config.vocab_size,
        ngram_block=config.ngram_block, algo_version=config.algo_version,
    )


def make_detector(config: WatermarkConfig, key: bytes, threshold: float | None = None) -> Detector:
    return Detector(config, key, threshold=threshold)
