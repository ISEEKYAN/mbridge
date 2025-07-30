# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


from ..core import register_model
from .qwen2 import Qwen2Bridge


@register_model("mimo")
class MimoBridge(Qwen2Bridge):
    """
    Bridge implementation for Mimo models.

    This class extends Qwen2Bridge to provide specific configurations and
    optimizations for Mimo models, handling the conversion between
    Hugging Face Mimo format and Megatron-Core.
    
    TODO: MTP layer is still WIP.
    """

