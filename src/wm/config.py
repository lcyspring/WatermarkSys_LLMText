"""算法与生成配置：统一记录、计算 config_hash，保证实验可追溯。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass
class WatermarkConfig:
    """单次生成/检测所使用的全部可变配置。

    config_hash 绑定模型、tokenizer、算法参数与密钥标识，
    使任意图表结论都能回溯到具体配置快照。
    """

    algo: str = "kgw"
    algo_version: str = "kgw-v1"
    gamma: float = 0.25                 # 绿名单比例（固定）
    delta: float = 0.0                 # logit 偏置强度；0 表示真正无水印基线
    context_width: int = 4             # 上下文宽度 h（固定）
    prf: str = "HMAC-SHA256"           # 伪随机函数（固定）
    ngram_block: int = 4               # 重复 n-gram 屏蔽窗口
    key_id: str = "default"            # 仅记录标识，不保存密钥本身

    vocab_size: int = 0
    model_id: str = ""
    tokenizer_revision: str = ""
    model_revision: str = ""
    dtype: str = "fp16"
    device: str = "cpu"

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 256
    seed: int = 0

    def config_hash(self) -> str:
        """配置快照的短哈希（16 位十六进制），用于数据血缘与一致性检查。"""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WatermarkConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})
