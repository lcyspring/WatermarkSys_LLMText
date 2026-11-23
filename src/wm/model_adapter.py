"""模型与 tokenizer 适配：延迟导入 torch/transformers，保证无 torch 环境也可导入。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelInfo:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dtype: str
    device: str
    vocab_size: int = 0


class ModelAdapter:
    """加载公开小模型并注册 logits processor。

    开题阶段最小验证：加载模型 → 注册 KGWLogitsProcessor → 生成一段带水印文本。
    所有重量级依赖在调用时才导入，便于在 CI/断网环境下做核心算法单测。
    """

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
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
            vocab_size=self._model.config.vocab_size,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def generate(self, prompt: str, logits_processor=None, **gen_kwargs) -> tuple[str, Any]:
        torch, _, _ = self._lazy()
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        params = dict(
            max_new_tokens=gen_kwargs.get("max_new_tokens", 256),
            do_sample=gen_kwargs.get("do_sample", True),
            temperature=gen_kwargs.get("temperature", 1.0),
            top_p=gen_kwargs.get("top_p", 1.0),
        )
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            params["logits_processor"] = LogitsProcessorList([logits_processor])
        with torch.no_grad():
            out = self._model.generate(**inputs, **params)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, new_tokens
