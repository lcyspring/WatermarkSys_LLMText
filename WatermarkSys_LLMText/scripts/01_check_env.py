"""环境与依赖自检（W1/W2 交付物）。"""
from __future__ import annotations

import importlib.util
import platform
import sys

from common import CODE_ROOT, ensure_dirs

OPTIONAL = ["torch", "transformers", "fastapi", "uvicorn", "jinja2",
            "numpy", "yaml", "pytest", "bert_score", "nltk"]


def main() -> None:
    ensure_dirs()
    print("== 环境自检 ==")
    print(f"  Python      : {sys.version.split()[0]} ({platform.machine()})")
    print(f"  代码根目录  : {CODE_ROOT}")
    print(f"  平台        : {platform.platform()}")
    print("\n== 依赖 ==")
    missing = []
    for m in OPTIONAL:
        ok = importlib.util.find_spec(m) is not None
        print(f"  {'OK ' if ok else 'MISS'} {m}")
        if not ok:
            missing.append(m)
    if "torch" in [m for m in OPTIONAL if importlib.util.find_spec(m)]:
        import torch
        print(f"\n  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if missing:
        print(f"\n[提示] 缺失可选依赖: {', '.join(missing)}")
        print("       纯算法与合成预演不需要 torch/transformers；")
        print("       真实模型实验需先安装并准备权重。")
    print("\n== 结论 ==")
    core_ok = importlib.util.find_spec("numpy") is not None
    print(f"  核心链路（纯 Python 算法 + 合成模拟）: "
          f"{'可用' if core_ok else '不可用'}")
    hf_ok = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))
    print(f"  真实模型链路: {'可用（需权重）' if hf_ok else '不可用'}")


if __name__ == "__main__":
    main()
