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
from mbridge import AutoBridge

# Load a model from Hugging Face
bridge = AutoBridge.from_pretrained("/path/to/Qwen/Qwen2.5-7B-Instruct")

# Get a Megatron-Core model
model = bridge.get_model()

# Generate with the model
# See examples/0.load_model_and_generate_single_gpu.py for detailed usage

# Export weights back to Hugging Face format
for key, weight in bridge.export_weights(model):
    # Process or save the exported weights
    print(f"Exported: {key}")
```

## Supported Models

Currently supported models:
- Qwen2
- Qwen2-MoE
- Qwen3
- Qwen3-MoE
- LLaMA2
- DeepseekV3

## Feature Highlights

- **Comprehensive Model Support**: Support for various model architectures including MoE models (Mixture of Experts)
- **Parameter Export**: Export trained Megatron parameters back to Hugging Face format
- **Easy-to-use API**: Simplified interface for model conversion and weight loading
- **Distributed Training Support**: Interface with Megatron-Core's advanced parallelism capabilities

## Examples

The `example` directory contains scripts demonstrating common use cases:

- `0.load_model_and_generate_single_gpu.py`: Loading a model and generating text
- `1.load_model_and_export_single_gpu.py`: Loading a model and exporting weights

## TODO

### 1. Testing with Different Parallelism Strategies
- [ ] Comprehensive testing with Tensor Parallelism (TP)
- [ ] Testing with Pipeline Parallelism (PP)
- [ ] Testing with Expert Parallelism (EP)
- [ ] Testing with combined parallelism strategies (TP+PP+EP)
- [ ] Benchmark performance across different parallelism configurations

### 2. Additional Examples
- [ ] Supervised Fine-Tuning (SFT) example
- [ ] Continue pretraining example
- [ ] Multi-modal model example
- [ ] Inference optimization examples
- [ ] Integration with popular training frameworks

### 3. Correctness Verification Process
- [ ] Develop a systematic validation pipeline
- [ ] Add comparison tools for logits between HF and Megatron implementations
- [ ] Create regression tests for weight loading
- [ ] Implement validation tests for all supported parallelism modes
- [ ] Document verification procedures for contributors

### 4. How to Contribute
- [ ] Set up contribution guidelines
- [ ] Create templates for issue reporting and pull requests
- [ ] Document code style and testing requirements
- [ ] Establish the review process
- [ ] Provide guidance for adding support for new models

### 5. Online Export
- [x] Online export of Megatron model weights to inference engines
- [ ] Add example of online export to inference engine

### 6. Advanced Training Techniques
- [ ] Implement sequence packing for training
- [ ] Implement dynamic batching
- [ ] Support efficient mixed-precision training

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
