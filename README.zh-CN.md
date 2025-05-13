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

# 使用模型生成内容
# 详细用法请参见examples/0.load_model_and_generate_single_gpu.py

# 导出权重回Hugging Face格式
for key, weight in bridge.export_weights(model):
    # 处理或保存导出的权重
    print(f"已导出: {key}")
```

## 支持的模型

当前支持的模型：
- Qwen2
- Qwen2-MoE
- Qwen3
- Qwen3-MoE
- LLaMA2
- DeepseekV3

## 功能亮点

- **全面的模型支持**：支持多种模型架构，包括MoE（混合专家）模型
- **参数导出**：将训练后的Megatron参数导出回Hugging Face格式
- **易用的API**：简化的模型转换和权重加载接口
- **分布式训练支持**：对接Megatron-Core的高级并行化能力

## 示例

`example`目录包含展示常见用例的脚本：

- `0.load_model_and_generate_single_gpu.py`：加载模型并生成文本
- `1.load_model_and_export_single_gpu.py`：加载模型并导出权重

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
- [x] 在线导出Megatron模型权重到推理引擎
- [ ] 添加在线导出到推理引擎的示例

### 6. 高级训练技术
- [ ] 使用sequence packing进行训练
- [ ] 实现动态批处理
- [ ] 支持高效的混合精度训练

## 许可证

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. 