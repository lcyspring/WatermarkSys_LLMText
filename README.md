# WatermarkSys_LLMText

面向大语言模型生成文本的水印嵌入与鲁棒检测系统 —— 本科毕业设计项目（李春雨，202305567128）。

## 项目目标

复现 KGW 绿名单水印，构建一个可复现的系统：可控生成 → 水印嵌入 → 统计检测 → 批量文本后处理 → 指标评估 → 可视化报告。

结论严格限定为“是否检出指定算法、配置与密钥的水印证据”，不用于判断作者身份、内容真实性或跨模型有效性。

## 当前状态

- 阶段：W1（环境与文献调研）
- 已完成：任务书解析、开题报告、答辩手册、实施计划书、执行清单
- 进行中：模型试跑、KGW 主链路设计

## 文档

| 文档 | 说明 |
|---|---|
| [实施计划书](docs/项目实施计划书.docx) | 16 周任务、实验口径、资源核验、GitHub 工作流、验收标准 |
| [项目执行清单](docs/项目执行清单.md) | 112 天可勾选任务，每日提醒读取 |
| [进度日志](docs/进度日志.md) | 每日记录与周报 |
| [开题报告](docs/开题报告_面向大语言模型生成文本的水印嵌入与鲁棒检测系统_终稿.docx) | 立项依据与技术路线 |
| [答辩手册](docs/答辩大白话手册_面向大语言模型文本水印.docx) | 原理类比、实验数字、高频问答 |
| [任务书](docs/大语言模型文本水印嵌入与鲁棒检测_本科毕业设计任务书.docx) | 学校原始要求 |
| [第三方资源](THIRD_PARTY.md) | 参考仓库、许可证与使用边界 |

## 计划目录结构

```text
WatermarkSys_LLMText/
├─ README.md
├─ requirements.txt
├─ configs/              # 模型、算法、实验、密钥样例
├─ src/wm/               # model_adapter, watermark, detect, postprocess, eval, pipeline, db, web
├─ scripts/              # 00_smoke_test.py ... 09_export_report.py
├─ tests/
├─ data/                 # 不入库
├─ outputs/              # 不入库
├─ docs/
└─ papers/               # 不入库
```

## 快速开始（W5 后可用）

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/00_smoke_test.py
pytest -q
```

## 关键约束

- 主模型：Qwen2.5-0.5B-Instruct，固定 revision 与 dtype
- 参数：gamma=0.25，delta ∈ {0, 0.5, 1, 2}，h=4，HMAC-SHA256
- 阈值：仅用校准集 400 负例，目标 FPR=1%，三档共用
- 最低有效长度：32 token
- 密钥不入库，只记录 key_id

## 许可证

本项目代码尚未确定开源许可证；参考的第三方仓库清单与许可边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。在确定许可证前，默认保留所有权利。
