"""算法与生成配置：统一记录、计算 config_hash，保证实验可追溯。

支持从扁平 YAML 读取（优先 pyyaml，缺失时使用内置极简解析器），
便于在最小环境下加载 configs/*.yaml 而不引入额外强依赖。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields


def _parse_flat_yaml(text: str) -> dict:
    """极简 YAML 解析：仅支持 `key: value` 单层与 `#` 注释。"""
    out: dict = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if val == "":
            continue
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            try:
                out[key] = int(val)
            except ValueError:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        return flatten(data)
    except ImportError:
        return _parse_flat_yaml(text)


def flatten(d: dict) -> dict:
    """一层嵌套展开：{a: {b: 1}} → {"a.b": 1}；单层原样返回。"""
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2
        else:
            out[k] = v
    return out


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

    def fingerprint(self) -> dict:
        """参与一致性校验的关键字段（换模型/参数必须变化）。"""
        return {
            "algo": self.algo,
            "algo_version": self.algo_version,
            "gamma": self.gamma,
            "context_width": self.context_width,
            "prf": self.prf,
            "ngram_block": self.ngram_block,
            "key_id": self.key_id,
            "vocab_size": self.vocab_size,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WatermarkConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "WatermarkConfig":
        data = load_yaml(path)
        cfg = cls.from_dict(data)
        for k, v in overrides.items():
            if k in {f.name for f in fields(cls)} and v is not None:
                setattr(cfg, k, v)
        return cfg
