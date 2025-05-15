# MBridge: 连接Megatron-Core与Hugging Face/强化学习

MBridge提供了Hugging Face模型和Megatron-Core优化实现之间的无缝桥接，用于高效的分布式训练和推理。同时，MBridge还提供了强化学习（RL）接入Megatron所需的必要工具和流程。

[English Documentation](README.md)

## 概述

MBridge允许您将流行的Hugging Face模型转换为Megatron-Core格式，使您能够利用先进的并行策略进行大规模训练和推理。该库支持各种模型架构，并简化了这些框架之间的转换过程。对于强化学习工作流，MBridge提供了连接RL算法与Megatron优化模型所需的接口和工具。

## 安装

```bash
# pip install mbridge
pip install -e .
```

## 快速开始

```python
from megatron.core import parallel_state as mpu
from mbridge import AutoBridge

# 初始化分布式环境
mpu.initialize_model_parallel(
    tensor_model_parallel_size=tp,
    pipeline_model_parallel_size=pp,
    virtual_pipeline_model_parallel_size=vpp,
    context_parallel_size=cp,
    expert_model_parallel_size=ep,
)

# 从Hugging Face加载模型
HF_MODEL_PATH = "/path/to/Qwen/Qwen2.5-7B-Instruct"
bridge = AutoBridge.from_pretrained(HF_MODEL_PATH)

# 获取Megatron-Core模型并从Hugging Face加载权重
model = bridge.get_model(weight_path=HF_MODEL_PATH)

# 导出权重回Hugging Face格式用于推理引擎
for key, weight in bridge.export_weights(model):
    # 处理或保存导出的权重
    print(f"已导出: {key}")
```

## 支持的模型

当前支持的模型：
- [x] Qwen2
- [x] Qwen2-MoE
- [x] Qwen3
- [x] Qwen3-MoE
- [x] LLaMA2
- [ ] DeepseekV3
- [ ] Mixtral

## 功能亮点

- **全面的模型支持**：支持多种模型架构，包括MoE（混合专家）模型
- **在线权重导入**：支持在线加载HF权重，支持各种并行策略，无需保存额外的Megatron-Core格式权重
- **在线权重导出**：支持在线导出权重到HF格式用于推理引擎，支持TP/PP/CP/VPP/EP/ETP等并行策略
- **分布式训练支持**：无缝对接Megatron-Core的高级并行化能力
- **简洁API**：直观的模型转换和权重管理接口

## 示例

`example`目录包含展示常见用例的脚本：

- `0.load_model_and_generate_single_gpu.py`：在单GPU上加载模型并生成文本
- `1.load_model_and_export_single_gpu.py`：在单GPU上加载模型并导出权重
- `2.load_model_and_export_multiple_gpus.py`：使用多个GPU（TP/PP/CP/VPP并行）加载模型并导出权重

## 开发路线图

### 功能
- [ ] VLM（视觉语言模型）支持
- [ ] FP8精度支持

### 更多示例
- [ ] 监督微调（SFT）示例
- [ ] 继续预训练示例
- [ ] 多模态模型示例
- [ ] 在线导出到推理引擎示例
- [ ] 与流行训练框架的集成

### 正确性验证流程
- [ ] 开发系统验证流程
- [ ] 添加HF与Megatron实现之间的logits比较工具
- [ ] 为权重加载创建回归测试
- [ ] 为所有支持的并行模式实现验证测试
- [ ] 为贡献者提供验证程序文档

### 社区贡献
- [ ] 建立贡献指南
- [ ] 创建问题报告和拉取请求的模板
- [ ] 记录代码风格和测试要求
- [ ] 建立审查流程
- [ ] 提供添加新模型支持的指导

### 高级训练技术
- [ ] 实现序列打包（sequence packing）进行训练
- [ ] 实现动态批处理

## 许可证

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. 