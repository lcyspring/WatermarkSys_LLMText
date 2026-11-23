"""模型适配层。

两种模式：
1. ``ModelAdapter``：真实 Hugging Face 因果语言模型（延迟导入 torch/transformers）。
2. ``SyntheticLM``：机制级离线模拟器 —— 用与 KGW 完全相同的绿名单偏置机制
   在合成 logits 上采样，用于断网/无权重环境下贯通“生成 → 检测 → 后处理 →
   校准 → 评估 → 报告”全流程，并对 delta 做敏感性预演。

重要边界：SyntheticLM 的结果只能证明**流水线与统计口径正确**，
不能替代真实模型实验；论文与答辩中必须标注 run 的 mode 字段。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from wm.config import WatermarkConfig
from wm.watermark.kgw import green_list


@dataclass
class ModelInfo:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dtype: str
    device: str
    vocab_size: int = 0
    mode: str = "hf"


# --------------------------------------------------------------------------- #
# 真实模型
# --------------------------------------------------------------------------- #
class ModelAdapter:
    """加载公开小模型并注册 KGW logits processor（需 torch + transformers）。"""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        dtype: str = "fp16",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.device = device
        self._model = None
        self._tokenizer = None

    def _lazy(self):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return torch, AutoModelForCausalLM, AutoTokenizer

    def load(self) -> ModelInfo:
        torch, AutoModelForCausalLM, AutoTokenizer = self._lazy()
        torch_dtype = torch.float16 if self.dtype == "fp16" else torch.float32
        if self.device:
            device = self.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=torch_dtype,
        ).to(device)
        return ModelInfo(
            model_id=self.model_id,
            model_revision=self.revision or "",
            tokenizer_revision=self.revision or "",
            dtype=self.dtype,
            device=device,
            vocab_size=int(self._model.config.vocab_size),
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def encode(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        logits_processor=None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> list[int]:
        torch, _, _ = self._lazy()
        from transformers import LogitsProcessorList
        device = next(self._model.parameters()).device
        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        procs = LogitsProcessorList([logits_processor]) if logits_processor else None
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                logits_processor=procs,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return out[0][input_ids.shape[1]:].tolist()


# --------------------------------------------------------------------------- #
# 机制级离线模拟器
# --------------------------------------------------------------------------- #
class SyntheticLM:
    """用真实 KGW 偏置机制驱动的合成语言模型。

    每一步：
      1. 由 numpy 生成 base logits ~ N(0, 1)（受 温度 缩放）；
      2. 对当前绿名单 token 加上 delta（与 KGWLogitsProcessor 完全一致）；
      3. softmax 后采样下一个 token。

    因此 delta 对绿名单命中率的作用方向与真实模型一致：
      delta=0 → 期望命中率 γ（无偏置）
      delta↑ → 命中率单调上升
    """

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = int(vocab_size)

    def encode(self, text: str) -> list[int]:
        """把提示文本映射到确定的伪 token（同文本恒等）。"""
        out = []
        for word in text.split():
            h = 0
            for ch in word:
                h = (h * 131 + ord(ch)) % 1_000_003
            out.append(h % self.vocab_size)
        return out or [0]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(f"w{i}" for i in ids)

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        config: WatermarkConfig,
        key: bytes,
        *,
        max_new_tokens: int | None = None,
    ) -> list[int]:
        import numpy as np

        n_new = int(max_new_tokens or config.max_new_tokens)
        V = self.vocab_size
        h = config.context_width
        ids = [int(t) % V for t in prompt_ids]

        rng = np.random.default_rng(
            (abs(hash((tuple(ids), config.seed, config.delta))) % (2 ** 32))
        )
        temp = max(float(config.temperature), 1e-3)

        for _ in range(n_new):
            ctx = ids[-h:]
            logits = rng.normal(0.0, 1.0, size=V)
            if config.delta != 0.0:
                gl = green_list(key, ctx, h, config.gamma, V,
                                config.algo_version)
                if gl:
                    logits[np.fromiter(gl, dtype=np.int64, count=len(gl))] += config.delta
            logits /= temp
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            nxt = int(rng.choice(V, p=p))
            ids.append(nxt)
        return ids[len(prompt_ids):]

    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id="synthetic-lm",
            model_revision="mech-sim-v1",
            tokenizer_revision="mech-sim-v1",
            dtype="float32",
            device="cpu",
            vocab_size=self.vocab_size,
            mode="synthetic",
        )


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def expected_green_rate(delta: float, gamma: float, sigma: float = 1.0) -> float:
    """理论期望：绿名单概率占比（正态 logits 假设下用于文档说明）。"""
    z = delta / (math.sqrt(2.0) * sigma)
    return gamma * sigmoid(z) / (gamma * sigmoid(z) + (1 - gamma) * 0.5)
