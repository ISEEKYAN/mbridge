# MBridge: 连接Megatron-Core与Hugging Face/强化学习

MBridge提供了Hugging Face模型和Megatron-Core优化实现之间的无缝桥接，用于高效的分布式训练和推理。同时，MBridge还提供了强化学习（RL）接入Megatron所需的必要工具和流程。

[English Documentation](README.md)

## 概述

MBridge允许您将流行的Hugging Face模型转换为Megatron-Core格式，使您能够利用先进的并行策略进行大规模训练和推理。该库支持各种模型架构，并简化了这些框架之间的转换过程。对于强化学习工作流，MBridge提供了连接RL算法与Megatron优化模型所需的接口和工具。

## 安装

```bash
pip install mbridge
```

## 快速开始

```python
from mbridge import AutoBridge

# 从Hugging Face加载模型
bridge = AutoBridge.from_pretrained("/path/to/Qwen/Qwen2.5-7B-Instruct")

# 获取Megatron-Core模型
model = bridge.get_model()
```

## 支持的模型

当前支持的模型：
- Qwen2
- Qwen3
- LLaMA2
- DeepseekV3（即将推出）

## 待办事项

### 1. 不同并行策略下的测试
- [ ] 使用张量并行（TP）进行全面测试
- [ ] 使用流水线并行（PP）进行测试
- [ ] 使用专家并行（EP）进行测试
- [ ] 使用组合并行策略（TP+PP+EP）进行测试
- [ ] 在不同并行配置下基准性能测试

### 2. 更多示例
- [ ] 监督微调（SFT）示例
- [ ] 继续预训练示例
- [ ] 多模态模型示例
- [ ] 推理优化示例
- [ ] 与流行训练框架的集成

### 3. 正确性验证流程
- [ ] 开发系统验证流程
- [ ] 添加HF与Megatron实现之间的logits比较工具
- [ ] 为权重加载创建回归测试
- [ ] 为所有支持的并行模式实现验证测试
- [ ] 为贡献者提供验证程序文档

### 4. 如何贡献
- [ ] 建立贡献指南
- [ ] 创建问题报告和拉取请求的模板
- [ ] 记录代码风格和测试要求
- [ ] 建立审查流程
- [ ] 提供添加新模型支持的指导

### 5. 在线导出
- [ ] 在线导出megatron模型权重到推理引擎
- [ ] 支持不同推理框架的导出格式
- [ ] 提供轻量级推理优化选项

### 6. 高级训练技术
- [ ] 使用sequence packing进行训练
- [ ] 实现动态批处理
- [ ] 支持高效的混合精度训练

## 许可证

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. 