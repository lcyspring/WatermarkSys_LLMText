"""一键跑完整离线预演（W7—W15 的流水线入口）。

    python scripts/12_run_all.py

依次执行：环境自检 → 配置冻结 → 提示集 → 校准 → 生成 → 检测 →
扰动 → 指标 → 压力 → 报告。任一步失败即停止并给出非零退出码。
"""
from __future__ import annotations

import subprocess
import sys
import time

from common import CODE_ROOT

STEPS = [
    ("环境自检", "01_check_env.py", []),
    ("配置冻结", "02_freeze_config.py", []),
    ("构建提示集", "03_make_prompts.py", []),
    ("阈值校准", "05_calibrate.py", []),
    ("批量生成", "04_generate.py", ["--mode", "synthetic", "--fresh"]),
    ("批量检测", "06_detect.py", []),
    ("后处理扰动", "07_edits.py", []),
    ("指标汇总", "08_eval.py", []),
    ("压力测试", "09_stress.py", []),
    ("质量与总报告", "10_report.py", []),
    ("盲评材料", "13_blind_eval.py", []),
]


def main() -> int:
    t0 = time.time()
    for i, (name, script, extra) in enumerate(STEPS, 1):
        print(f"\n{'='*70}\n[{i}/{len(STEPS)}] {name}  ({script})\n{'='*70}")
        cmd = [sys.executable, str(CODE_ROOT / "scripts" / script)] + extra
        r = subprocess.run(cmd, cwd=str(CODE_ROOT / "scripts"))
        if r.returncode != 0:
            print(f"\n[失败] {name} 返回码 {r.returncode}")
            return r.returncode
    print(f"\n{'='*70}\n全部完成，用时 {time.time()-t0:.1f}s")
    print(f"报告目录: {CODE_ROOT.parent / 'docs' / 'reports'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
