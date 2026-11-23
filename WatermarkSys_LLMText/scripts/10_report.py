"""质量评估与总报告（W10/W15 交付物）。

计算各 delta 档的质量代理指标（distinct-n / 重复率 / 压缩比），
并把校准、主结果、鲁棒性、压力测试汇总为一份总报告。
"""
from __future__ import annotations

import argparse
import json

from common import (METRICS_PATH, OUT, REPORTS, ensure_dirs, markdown_table,
                    read_jsonl)
from wm.eval.quality import quality_report


def _load(path):
    try:
        return json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    ensure_dirs()

    gens = [g for g in read_jsonl(OUT / "generations.jsonl")
            if g.get("status") == "completed"]
    if not gens:
        raise SystemExit("缺少生成数据，请先运行 04_generate.py")

    by_delta = {}
    for g in gens:
        by_delta.setdefault(float(g["delta"]), []).append(g.get("text", ""))

    qrows, qjson = [], {}
    for d in sorted(by_delta):
        q = quality_report(by_delta[d])
        qjson[str(d)] = q
        qrows.append([d, q["count"], q["len_mean"], q["distinct1_mean"],
                      q["distinct2_mean"], q["repeat4_mean"],
                      q["compression_mean"]])

    from common import write_json
    write_json(qjson, OUT / "quality_summary.json")

    qlines = [
        "# 质量评估报告（W10）", "",
        "指标均为无模型代理指标（离线可算）：",
        "",
        "- distinct-1 / distinct-2：n-gram 多样性，越高越多样；",
        "- repeat-4：重复 4-gram 占比，越低越好；",
        "- compression：zlib 压缩比，越高说明文本越冗余。",
        "",
        "> 说明：水印应尽量不损害文本质量。代理指标只反映表层统计特性，",
        "> 真实语义质量需 BERTScore / 人工盲评（W14），当前环境不联网故未执行。",
        "",
        markdown_table(["delta", "样本数", "平均词数", "distinct-1",
                        "distinct-2", "repeat-4", "压缩比"], qrows),
        "",
        "## 解读", "",
        "1. 若各 delta 档指标接近，说明偏置强度未显著改变表层多样性；",
        "2. 压缩比随 delta 上升而明显升高时，应警惕重复文本增多；",
        "3. 合成预演下的文本为伪词序列，指标仅用于回归监测。",
        "",
    ]
    (REPORTS / "质量评估报告.md").write_text("\n".join(qlines) + "\n",
                                            encoding="utf-8")

    calib = _load(OUT / "threshold_v1.json")
    metrics = _load(METRICS_PATH)
    stress = _load(OUT / "stress_results.json")
    freeze = _load(OUT / "config_freeze.json")
    edits = read_jsonl(OUT / "edit_detections.jsonl")

    main_rows = []
    for k, v in (metrics.get("by_delta") or {}).items():
        main_rows.append([k, v.get("n"), v.get("z_mean"),
                          f"{v.get('detected_rate', 0):.2%}"
                          if isinstance(v.get("detected_rate"), (int, float))
                          else v.get("detected_rate"),
                          v.get("fpr"), v.get("fnr"), v.get("auc")])

    rob_rows = [[r["edit_type"], r["level"], r["n_variants"], r["n_mean"],
                 r["z_mean"], f"{r['detected_rate']:.2%}"]
                for r in metrics.get("robustness", [])]

    lines = [
        "# 离线预演总报告（W1—W16 汇总）", "",
        "## 0. 本次运行的性质", "",
        "本次运行使用 **机制级离线模拟器**（SyntheticLM，V=2048）驱动完整流水线：",
        "生成 → 检测 → 校准 → 后处理扰动 → 指标 → 报告。",
        "",
        "它验证了：绿名单划分、logits 偏置、z/p 统计量、阈值校准、",
        "分层评估、鲁棒性度量、配置血缘的可运行性与口径一致性。",
        "",
        "它 **不能** 代替真实模型实验：真实语义质量、真实攻击鲁棒性、",
        "真实 FPR 需在有 GPU 与模型权重的环境中重跑（脚本与口径不变）。",
        "",
        "## 1. 配置快照", "",
        f"- config_hash = `{freeze.get('config_hash', 'N/A')}`",
        f"- freeze_digest = `{freeze.get('freeze_digest', 'N/A')}`",
        f"- Python = {freeze.get('python', 'N/A')}",
        "- 固定参数：gamma=0.25，h=4，HMAC-SHA256，ngram_block=4，min_tokens=32",
        "",
        "## 2. 阈值校准", "",
        f"- 阈值 z ≥ **{calib.get('threshold', 'N/A')}**"
        f"（{calib.get('threshold_id', '')}）",
        f"- 负例数 {calib.get('n_neg', 'N/A')}，目标 FPR "
        f"{calib.get('target_fpr', 'N/A')}，经验 FPR {calib.get('emp_fpr', 'N/A')}",
        f"- Wilson 95% CI = {calib.get('fpr_ci', 'N/A')}",
        "",
        "## 3. 主结果", "",
        markdown_table(["delta", "样本数", "平均 z", "检出率", "FPR", "FNR", "AUC"],
                       main_rows),
        "",
        "## 4. 后处理鲁棒性", "",
        markdown_table(["扰动", "档位", "变体数", "平均 n", "平均 z", "检出率"],
                       rob_rows) if rob_rows else "（尚未运行 07_edits.py）",
        "",
        "## 5. 压力与一致性", "",
        f"- 正确密钥平均 z：{stress.get('key_test', {}).get('正确密钥', 'N/A')}",
        f"- 错误密钥平均 z：{stress.get('key_test', {}).get('错误密钥', 'N/A')}",
        f"- gamma 不一致：{stress.get('gamma_test', 'N/A')}",
        f"- 温度敏感性：{stress.get('temperature_test', 'N/A')}",
        "",
        "## 6. 质量代理指标", "",
        markdown_table(["delta", "样本数", "平均词数", "distinct-1",
                        "distinct-2", "repeat-4", "压缩比"], qrows),
        "",
        "## 7. 结论边界（必须写进论文与答辩）", "",
        "1. 系统只回答“**是否检出指定算法、配置与密钥的水印证据**”；",
        "2. 未检出 ≠ 人类撰写；检出 ≠ 内容真实；",
        "3. 阈值仅由负例确定，三个 delta 档共用，不允许按结果回调；",
        "4. 400 负例下零误报的置信区间上界仍约为 0.9%，不能宣称 FPR=0；",
        "5. 无密钥者无法读取水印（错误密钥下 z 回到 0 附近）。",
        "",
        "## 8. 复现命令", "",
        "```bash",
        "cd WatermarkSys_LLMText",
        "python scripts/12_run_all.py          # 一键离线预演",
        "python -m pytest tests -q             # 单元测试",
        "python scripts/11_web.py              # 启动演示系统",
        "```",
        "",
        f"- 生成记录 {len(gens)} 条；检测记录 "
        f"{metrics.get('n_detections', 'N/A')} 条；扰动变体 {len(edits)} 条",
    ]
    (REPORTS / "总报告_离线预演.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    print(f"[质量] 各档指标: {json.dumps(qjson, ensure_ascii=False)[:200]}...")
    print(f"[总报告] {REPORTS / '总报告_离线预演.md'}")
    print(f"[质量报告] {REPORTS / '质量评估报告.md'}")


if __name__ == "__main__":
    main()
