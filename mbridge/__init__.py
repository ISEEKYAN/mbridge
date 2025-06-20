# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
MBridge: A bridge between Hugging Face models and Megatron-Core.

MBridge provides tools and classes to convert and run Hugging Face models
using Megatron-Core for efficient distributed training and inference.

The package contains:
- Core bridge classes for different model types
- Model-specific implementations for popular architectures
- Utilities for handling parallel states and transformations

Version: 0.1.0
"""

__version__ = "0.1.0"

# Import models module to ensure registration decorators are executed
from . import models, utils

# Export core classes
from .core.auto_bridge import AutoBridge
