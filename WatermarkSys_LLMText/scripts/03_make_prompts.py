"""构建提示集（W3 交付物）。

60 条提示，覆盖 4 类任务 × 2 种语言，编号固定、可复现；
split 字段用于区分校准集（calib）与测试集（test）。
"""
from __future__ import annotations

import json

from common import PROMPTS, ensure_dirs, write_jsonl

TOPICS = [
    ("学术", "用 200 字说明大语言模型文本水印的基本原理，并指出其局限。"),
    ("学术", "解释假设检验中 p 值的含义，说明为什么 p 值小不等于结论一定为真。"),
    ("学术", "比较绿名单水印与基于语义的水印在鲁棒性上的差异。"),
    ("学术", "简述重复检测（repetition detection）为什么会影响水印统计量。"),
    ("学术", "说明为什么水印检测不能证明文本由人类撰写。"),
    ("学术", "讨论温度参数如何影响采样的多样性与水印检出率。"),
    ("学术", "解释 HMAC 与普通哈希在水印密钥管理上的差别。"),
    ("学术", "说明为什么阈值必须用独立的负例集合校准。"),
    ("学术", "描述误报率（FPR）与漏报率（FNR）的权衡关系。"),
    ("学术", "简述长度对检测统计量功效的影响。"),
    ("技术", "用 Python 说明如何对一个列表做去重并保持顺序。"),
    ("技术", "解释 Git 中 rebase 与 merge 的区别及适用场景。"),
    ("技术", "说明 REST 与 GraphQL 在接口设计上的主要差异。"),
    ("技术", "解释数据库索引为什么能加速查询，代价是什么。"),
    ("技术", "描述 FastAPI 中依赖注入的基本用法。"),
    ("技术", "说明单元测试与集成测试的边界与分工。"),
    ("技术", "解释 SQLite 适合与不适合的场景。"),
    ("技术", "说明日志与指标在可观测性中的不同作用。"),
    ("技术", "解释幂等性为什么对重试机制很重要。"),
    ("技术", "描述 JWT 的基本结构与常见安全风险。"),
    ("新闻", "写一段关于城市公共交通优化的新闻导语，约 120 字。"),
    ("新闻", "为一则高校科研进展撰写简讯，要求客观、含数据。"),
    ("新闻", "写一段关于社区养老服务的新闻背景介绍。"),
    ("新闻", "为一次环保志愿活动写 150 字新闻稿。"),
    ("新闻", "简述一则科技产品发布会的新闻要点。"),
    ("新闻", "写一段关于乡村教育变化的报道开头。"),
    ("新闻", "为一则体育赛事写 100 字赛况。"),
    ("新闻", "写一段关于新能源产业的政策解读。"),
    ("新闻", "为一则医疗研究成果撰写科普式报道。"),
    ("新闻", "写一段关于城市夜经济的观察性报道。"),
    ("问答", "为什么天空是蓝色的？请给出通俗解释。"),
    ("问答", "如何在一周内提高英语听力？给出可执行计划。"),
    ("问答", "感冒和流感有什么区别？"),
    ("问答", "怎样判断一段文本是否由 AI 生成？"),
    ("问答", "什么是零知识证明？举一个生活化例子。"),
    ("问答", "如何为毕业设计安排时间表？"),
    ("问答", "为什么有些电池在低温下续航下降？"),
    ("问答", "怎样写一份清晰的实验报告？"),
    ("问答", "解释一下什么是过拟合，以及如何缓解。"),
    ("问答", "如何评估一个水印方案是否可用？"),
    ("Academic", "Explain in about 150 words how a green-list watermark works."),
    ("Academic", "Describe the difference between false positives and false negatives."),
    ("Academic", "Why should a detection threshold be calibrated on negatives only?"),
    ("Academic", "Summarize the role of the pseudorandom function in watermarking."),
    ("Academic", "Explain why watermark detection is not authorship attribution."),
    ("Technical", "Show how to implement a retry with exponential backoff."),
    ("Technical", "Explain the difference between latency and throughput."),
    ("Technical", "Describe how to structure a Python package for experiments."),
    ("Technical", "Explain why reproducible seeds matter in machine learning."),
    ("Technical", "Describe how to log experiment metadata for traceability."),
    ("News", "Write a 120-word news lead about a new public library opening."),
    ("News", "Write a short report on a university robotics competition."),
    ("News", "Write a brief news summary about renewable energy growth."),
    ("News", "Write a 100-word bulletin about a local marathon."),
    ("News", "Write a short piece on a city's air quality improvement."),
    ("QA", "How can I tell if a paragraph was written by an AI?"),
    ("QA", "What is a p-value in simple terms?"),
    ("QA", "How do I design a fair evaluation for a text watermark?"),
    ("QA", "What are the trade-offs of paraphrasing watermarked text?"),
    ("QA", "How should I report confidence intervals for a small sample?"),
]


def main() -> None:
    ensure_dirs()
    rows = []
    for i, (task, text) in enumerate(TOPICS, start=1):
        pid = f"P{i:04d}"
        split = "calib" if i <= 40 else "test"
        rows.append({
            "prompt_id": pid,
            "task_type": task,
            "lang": "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "en",
            "split": split,
            "text": text,
            "source": "authored_for_project",
            "revision": "v1",
        })
    write_jsonl(rows, PROMPTS)
    print(f"[提示集] {PROMPTS}  共 {len(rows)} 条")
    from collections import Counter
    print("  任务分布:", dict(Counter(r["task_type"] for r in rows)))
    print("  划分分布:", dict(Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()
