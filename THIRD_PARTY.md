# 第三方资源与许可记录

核验日期：2026-10-21。星标数只作活跃度参考，不作为质量依据。

| 资源 | 仓库 | 许可证 | 星标 | 最近更新 | 用途 | 使用边界 |
|---|---|---|---|---|---|---|
| KGW 官方实现 | jwkirchenbauer/lm-watermarking | Apache-2.0 | 704 | 2026-10-11 | 学习 logits processor、绿名单、重复 n-gram | 独立实现核心代码，复制处需注明 |
| Unigram 官方实现 | XuandongZhao/Unigram-Watermark | MIT | 39 | 2026-06-24 | 选做第二算法 | 可复用，需注明来源 |
| MarkLLM | THU-BPM/MarkLLM | Apache-2.0 | 1089 | 2026-10-20 | 参考评测接口与攻击组织 | 不直接套壳作为毕设系统 |
| SIR 语义水印 | THU-BPM/Robust_Watermark | 未声明 | 39 | 2024-11-13 | 理解语义水印思路 | 不复制代码、不打包分发 |
| WaterBench | THU-KEG/WaterBench | MIT | 33 | 2026-08-11 | 参考任务与评测协议 | 数据子集单独核许可 |
| X-SIR | zwhe99/X-SIR | 未声明 | 49 | 2026-08-10 | 阅读参考 | 研究代码，不复制 |
| waterpipe | si-mon-jinn/waterpipe | EUPL-1.2 | 0 | — | 参考流水线架构 | copyleft，不复制代码 |
| sbw | si-mon-jinn/sbw | EUPL-1.2 | 0 | — | 了解 vLLM logits processor | copyleft，不复制代码 |
| unwatermark | TheElderWyrm/unwatermark | MIT | 0 | — | 参考攻击实现 | 研究代码，需自测 |
| MarkMyWords | wagner-group/MarkMyWords | Apache-2.0 | 32 | 2024-06-06 | 参考评测设计 | 更新较旧，不作主依赖 |
| SynthID-Text | google-deepmind/synthid-text | Apache-2.0 | 1186 | 2026-09-30 | 拓展阅读 | 不纳入主实验 |

## 未采用或未核实的资源

- “LUT 大学 2025 本科毕设（SIR 自适应水印）”：多次检索未找到可公开访问的原文，不作为必读或引用。
- “THU-BPM/WaterPark”：GitHub 上未检索到对应仓库，不引用、不依赖。
- “X-SIR 为入门级教学实现”：该说法不准确，实为研究代码且未声明许可证。

## 记录规范

每引用一个外部仓库，在本文件增加：仓库、commit SHA、LICENSE、参考的文件或函数、是否复制代码、个人改动。代码文件仅在确实参考具体实现时注明来源。
