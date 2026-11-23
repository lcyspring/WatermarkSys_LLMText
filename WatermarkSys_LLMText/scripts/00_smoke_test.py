"""最小冒烟测试：在无 torch 环境下验证核心链路自洽。

演示内容（对应开题报告“最小链路验证”的算法部分）：
1. 配置 → config_hash 绑定；
2. 构造全绿序列 → 检测 z 显著为正；
3. 均匀随机文本 → 检测 z 接近 0；
4. 阈值校准（合成负例）→ 经验阈值；
5. 评估指标计算口径；
6. 三类后处理可运行并记录实际强度。

运行：python scripts/00_smoke_test.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wm.config import WatermarkConfig  # noqa: E402
from wm.detect import calibrate_threshold, fpr_binomial_ci  # noqa: E402
from wm.eval import confusion  # noqa: E402
from wm.postprocess import delete_words, replace_synonyms, truncate  # noqa: E402
from wm.watermark import detect, green_list  # noqa: E402

KEY = b"smoke-test-key"
V = 1000
GAMMA = 0.25
H = 4


def build_all_green(length=80):
    seq = list(range(H))
    for _ in range(length - H):
        gl = green_list(KEY, seq[-H:], H, GAMMA, V)
        seq.append(min(gl))
    return seq


def main():
    cfg = WatermarkConfig(vocab_size=V)
    print("== 1. 配置快照 ==")
    print(f"  config_hash = {cfg.config_hash()} (delta={cfg.delta})")

    print("\n== 2. 全绿序列检测（模拟强水印） ==")
    seq = build_all_green(80)
    res = detect(seq, KEY, GAMMA, H, V, threshold=3.0, min_tokens=32)
    print(f"  status={res['status']} n={res['n']} g={res['g']} "
          f"z={res['z']:.2f} decision={res['decision']}")

    print("\n== 3. 随机文本检测（无水印负例） ==")
    rng = random.Random(7)
    rand = [rng.randrange(V) for _ in range(120)]
    res2 = detect(rand, KEY, GAMMA, H, V, min_tokens=32)
    print(f"  status={res2['status']} n={res2['n']} z={res2['z']:.2f}")

    print("\n== 4. 阈值校准（合成 400 负例） ==")
    neg_z = []
    for i in range(400):
        r = random.Random(1000 + i)
        s = [r.randrange(V) for _ in range(120)]
        neg_z.append(detect(s, KEY, GAMMA, H, V, min_tokens=32)["z"])
    tau = calibrate_threshold(neg_z, target_fpr=0.01)
    fp = sum(1 for z in neg_z if z >= tau)
    lo, hi = fpr_binomial_ci(fp, len(neg_z))
    print(f"  threshold(z)={tau:.3f}  emp_fpr={fp}/{len(neg_z)}={fp/len(neg_z):.4f}  "
          f"95%CI=[{lo:.4f},{hi:.4f}]")

    print("\n== 5. 评估指标口径 ==")
    y_true = [True] * 100 + [False] * 100
    y_pred = [True] * 90 + [False] * 10 + [True] * 5 + [False] * 95
    m = confusion(y_true, y_pred)
    print(f"  {m.to_dict()}")

    print("\n== 6. 三类后处理 ==")
    text = "the good big small happy said good bad big small happy"
    d = delete_words(text, 0.3, seed=1)
    t = truncate(text, 0.5)
    s = replace_synonyms(text, 0.3, seed=2)
    print(f"  delete : actual_rate={d.actual_rate:.2f} len {d.original_count}->{d.edited_count}")
    print(f"  truncate: keep_ratio={t.kept_ratio:.2f} len {t.original_count}->{t.kept_count}")
    print(f"  synonym : replaced {s.replaced_count}/{s.candidate_count} "
          f"actual_rate={s.actual_rate:.2f}")

    print("\n冒烟测试通过：核心链路在无 torch 环境下自洽。")
    print("下一步：安装 torch/transformers 后用 scripts/02_generate.py 跑真实生成。")


if __name__ == "__main__":
    main()
