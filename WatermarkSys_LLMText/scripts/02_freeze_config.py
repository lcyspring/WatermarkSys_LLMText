"""冻结实验配置（W2/W4 交付物）：记录参数、依赖版本与 config_hash。"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys

from common import CODE_ROOT, METRICS_PATH, OUT, ensure_dirs, get_config, write_json


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60)
        return [l for l in out.stdout.splitlines()
                if l.split("==")[0].lower() in {
                    "numpy", "torch", "transformers", "fastapi", "uvicorn",
                    "jinja2", "pydantic", "pyyaml", "pytest"}]
    except Exception:
        return []


def main() -> None:
    ensure_dirs()
    base = get_config(delta=0.0)
    freeze = {
        "frozen_at": "2026-11-23",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": base.to_dict(),
        "config_hash": base.config_hash(),
        "fingerprint": base.fingerprint(),
        "dependencies": _pip_freeze(),
        "fixed_params": {
            "gamma": 0.25,
            "context_width_h": 4,
            "prf": "HMAC-SHA256",
            "delta_levels": [0.0, 0.5, 1.0, 2.0],
            "ngram_block": 4,
            "min_effective_tokens": 32,
            "target_fpr": 0.01,
        },
        "notes": [
            "delta=0 为真正无水印基线，生成端不修改任何 logits",
            "阈值只由校准集负例确定，三档 delta 共用",
            "未检出 ≠ 人类撰写；检出 ≠ 内容真实",
        ],
    }
    path = OUT / "config_freeze.json"
    write_json(freeze, path)
    digest = hashlib.sha256(
        json.dumps(freeze, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    freeze["freeze_digest"] = digest
    write_json(freeze, path)
    print(f"[配置冻结] {path}")
    print(f"  config_hash = {freeze['config_hash']}  freeze_digest = {digest}")
    for k, v in freeze["fixed_params"].items():
        print(f"  {k} = {v}")
    # 同时写一份到 docs 便于评审
    from common import REPORTS
    md = ["# 配置冻结记录", "",
          f"- 冻结时间：{freeze['frozen_at']}",
          f"- config_hash：`{freeze['config_hash']}`",
          f"- freeze_digest：`{digest}`",
          f"- Python：{freeze['python']}", "",
          "## 固定参数", ""]
    for k, v in freeze["fixed_params"].items():
        md.append(f"- {k} = {v}")
    md += ["", "## 依赖", ""] + [f"- {d}" for d in freeze["dependencies"]]
    (REPORTS / "配置冻结记录.md").write_text("\n".join(md) + "\n",
                                              encoding="utf-8")
    print(f"[文档] {REPORTS / '配置冻结记录.md'}")


if __name__ == "__main__":
    main()
