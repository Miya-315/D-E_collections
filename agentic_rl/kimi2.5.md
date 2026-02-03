```markdown
# Kimi K2.5 — Section 2.1 笔记（Native Multimodal Pre-Training）

## 一句话结论
- 在“固定总 vision+text token 预算”下，早期融合（early fusion）且较低的视觉比例比在训练后期大量注入视觉数据效果更好；因此 K2.5 在整个预训练过程保持恒定的视文本混合比。

## 实验设置（关键细节）
- 固定总 token 预算：通过先用纯文本训练若干 token 再引入视觉数据来实现不同“vision injection timing”与“vision-text ratio”。
- 对比策略：Early / Mid / Late 注入策略与不同视觉比例，评估多项指标（Vision Knowledge, Vision Reasoning, OCR, Text Knowledge, Text Reasoning, Code）。

## 表格与结果摘要
- 表 1 显示 Early (10% vision) 在多数视觉与文本指标上领先例：
	- Vision Reasoning: Early 43.8 vs Mid 40.7 vs Late 39.0
	- Text Knowledge: Early 45.5 vs Mid 43.9 vs Late 43.1

## 架构与实现要点
- 视觉编码器：MoonViT-3D，原生分辨率、NaViT patch packing，可处理可变分辨率图像。
- 视频处理：连续 4 帧打包，patch 级时间平均，实现 4× 时间压缩并共享图像/视频编码权重。
- 训练流程：先单独训练 ViT（约 1T tokens），再 joint pre-training（约 15T vision-text tokens）。

## 作者的解释与直觉
- 早期低比例融合让模型有更长时间学习跨模态对齐与共同表示，避免后期突然注入大量视觉数据导致的“冲击”。
- 在 joint pretraining 已建立较好对齐的情形下，后续用纯文本 SFT（zero-vision SFT）也能激活视觉相关能力（见 Section 2.2）。

## 批判性思考 / 注意事项
- “固定总 token”实验的控制细节很重要：图像映射到 token 的方式、数据源质量与配比会显著影响结论。
- 需检查是否有重复试验与统计显著性（置信区间）；表中差距部分较小，需要谨慎解读。
- 早期融合优势可能依赖于具体架构（MoonViT-3D + K2 MoE）或训练数据分布，不一定对所有模型通用。

## 对我们学习的启发
- 在资源受限（固定 token 预算）场景下，优先早期且适度地混合视觉信息，可能更有利于跨模态对齐。
- 视觉编码器设计应优先考虑权重共享与时间压缩策略（如 4 帧打包 + patch pooling），以在同一模型中兼顾图像与视频能力。

```
#