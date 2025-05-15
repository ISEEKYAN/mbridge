# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from ..core.bridge import Bridge, register_model


@register_model("deepseekv3")
class DeepseekV3Bridge(Bridge):
    """
    Specific bridge implementation for DeepseekV3 models.
    """

    def get_model(self, load_weights=False):
        raise NotImplementedError("DeepseekV3 model bridge not yet implemented")
