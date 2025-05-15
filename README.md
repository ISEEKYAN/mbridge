# MBridge: Bridge Megatron-Core to Hugging Face/Reinforcement Learning

MBridge provides a seamless bridge between Hugging Face models and Megatron-Core's optimized implementation for efficient distributed training and inference. It also offers necessary tools and processes for integrating Reinforcement Learning (RL) with Megatron.

[中文文档](README.zh-CN.md)

## Overview

MBridge allows you to convert popular Hugging Face models to Megatron-Core format, enabling you to leverage advanced parallelism strategies for large-scale training and inference. The library supports various model architectures and simplifies the process of transitioning between these frameworks. For Reinforcement Learning workflows, MBridge provides interfaces and tools needed to connect RL algorithms with Megatron-optimized models.

## Installation

```bash
# pip install mbridge
pip install -e .
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
bridge = AutoBridge.from_pretrained(HF_MODEL_PATH)

# Get a Megatron-Core model and load weights from Hugging Face
model = bridge.get_model(weight_path=HF_MODEL_PATH)

# Export weights back to Hugging Face format for inference engine
for key, weight in bridge.export_weights(model):
    # Process or save the exported weights
    print(f"Exported: {key}")
```

## Supported Models

Currently supported models:
- [x] Qwen2
- [x] Qwen2-MoE
- [x] Qwen3
- [x] Qwen3-MoE
- [x] LLaMA2
- [ ] DeepseekV3
- [ ] Mixtral

## Feature Highlights

- **Comprehensive Model Support**: Support for various model architectures including MoE models (Mixture of Experts)
- **Online Weight Import**: Online loading HF weights with various parallelism strategies, no need to save extra Megatron-Core format weights
- **Online Weight Export**: Online exporting weights to HF format for inference engines, with support for TP/PP/CP/VPP/EP/ETP parallelism strategies
- **Distributed Training Support**: Seamless interface with Megatron-Core's advanced parallelism capabilities
- **Simple API**: Intuitive interfaces for model conversion and weight management

## Examples

The `example` directory contains scripts demonstrating common use cases:

- `0.load_model_and_generate_single_gpu.py`: Loading a model and generating text on a single GPU
- `1.load_model_and_export_single_gpu.py`: Loading a model and exporting weights on a single GPU
- `2.load_model_and_export_multiple_gpus.py`: Loading a model and exporting weights using multiple GPUs with TP/PP/CP/VPP parallelism

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
