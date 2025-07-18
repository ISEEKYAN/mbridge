# MBridge: Bridge Megatron-Core to Hugging Face/Reinforcement Learning

MBridge provides a seamless bridge between Hugging Face models and Megatron-Core's optimized implementation for efficient distributed training and inference. It also offers necessary tools and processes for integrating Reinforcement Learning (RL) with Megatron.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ISEEKYAN/mbridge)
[中文文档](README.zh-CN.md)

## Overview

MBridge allows you to convert popular Hugging Face models to Megatron-Core format, enabling you to leverage advanced parallelism strategies for large-scale training and inference. The library supports various model architectures and simplifies the process of transitioning between these frameworks. For Reinforcement Learning workflows, MBridge provides interfaces and tools needed to connect RL algorithms with Megatron-optimized models.

## Feature Highlights

- **Comprehensive Model Support**: Support for various model architectures including MoE models (Mixture of Experts)
- **Online Weight Import**: Online loading HF weights with various parallelism strategies, auto shard the weights, no need to save extra Megatron-Core format weights
- **Online Weight Export**: Online exporting weights to HF format for inference engines, with support for TP/PP/CP/VPP/EP/ETP parallelism strategies
- **Memory Friendly**: Use per-tensor strategies, minimize the memory peak when loading/exporting HF format weights.
- **Simple API**: Intuitive interfaces for model conversion and weight management
- **Support Transformer Engine**: Use the powerful Transformer Engine to accelerate Megatron-Core models for better performance (use_te=False is not supported now)

## Installation

```bash
pip install mbridge
```

## Quick Start

```python
from megatron.core import parallel_state as mpu
from mbridge import AutoBridge

# Initialize distributed environment
mpu.initialize_model_parallel(
    tensor_model_parallel_size=tp,
    pipeline_model_parallel_size=pp,
    virtual_pipeline_model_parallel_size=vpp,
    context_parallel_size=cp,
    expert_model_parallel_size=ep,
)

# Load a model from Hugging Face
HF_MODEL_PATH = "/path/to/Qwen/Qwen2.5-7B-Instruct"
# or llama model
HF_MODEL_PATH = "/path/to/llama/llama3-8b-instruct"
bridge = AutoBridge.from_pretrained(HF_MODEL_PATH)

# Get a Megatron-Core model and load weights from Hugging Face
model = bridge.get_model(weight_path=HF_MODEL_PATH)

# Export weights back to Hugging Face format for inference engine
for key, weight in bridge.export_weights(model):
    # Process or save the exported weights
    print(f"Exported: {key}")

# save model with HF format
bridge.save_weights(model, "path/to/save/model", memory_efficient=False) # set memory_efficient=True if the model is vary large
```

## Supported Models

Currently supported models:
- [x] Qwen2
- [x] Qwen2-MoE
- [x] Qwen3
- [x] Qwen3-MoE
- [x] LLaMA
- [x] DeepseekV3
- [x] Mixtral
- [x] Qwen2.5-VL


## Examples

The `example` directory contains scripts demonstrating common use cases:

- `0.load_model_and_generate_single_gpu.py`: Loading a model and generating text on a single GPU
- `1.load_model_and_export_single_gpu.py`: Loading a model and exporting weights on a single GPU
- `2.load_model_and_export_multiple_gpus.py`: Loading a model and exporting weights using multiple GPUs with TP/PP/CP/VPP parallelism

### Post Model Creation Callbacks

MBridge provides a set of post model creation callbacks to customize the model after it is created.

- `make_value_model`: Add a value model to the model
- `freeze_moe_router`: Freeze the router of the model

```python
from mbridge.utils.post_creation_callbacks import make_value_model, freeze_moe_router

bridge = AutoBridge.from_pretrained(HF_MODEL_PATH)
model = bridge.get_model(weight_path=HF_MODEL_PATH, post_model_creation_callbacks=[make_value_model, freeze_moe_router])

```

## Development Roadmap

### Features
- [ ] VLM (Vision Language Model) support
- [ ] FP8 precision support

### Additional Examples
- [ ] Supervised Fine-Tuning (SFT) example
- [ ] Continue pretraining example
- [ ] Multi-modal model example
- [ ] Online export to inference engine example
- [ ] Integration with popular training frameworks

### Correctness Verification Process
- [ ] Develop a systematic validation pipeline
- [ ] Add comparison tools for logits between HF and Megatron implementations
- [ ] Create regression tests for weight loading
- [ ] Implement validation tests for all supported parallelism modes
- [ ] Document verification procedures for contributors

### Community Contribution
- [ ] Set up contribution guidelines
- [ ] Create templates for issue reporting and pull requests
- [ ] Document code style and testing requirements
- [ ] Establish the review process
- [ ] Provide guidance for adding support for new models

### Advanced Training Techniques
- [ ] Implement sequence packing for training
- [ ] Implement dynamic batching

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
