# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from ..core import LLMBridge, register_model


@register_model("qwen3")
class Qwen3Bridge(LLMBridge):
    """
    Bridge implementation for Qwen3 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen3 models, handling the conversion between
    Hugging Face Qwen3 format and Megatron-Core.
    """

    def _build_config(self):
        """
        Build the configuration for Qwen3 models.

        Configures Qwen3-specific parameters such as QK layer normalization.
        Qwen3 uses layer normalization on query and key tensors.

        Returns:
            TransformerConfig: Configuration object for Qwen3 models
        """
        return self._build_base_config(
            use_cpu_initialization=False,
            add_bias_linear=False,
            # qwen3
            qk_layernorm=True,
            **self.extra_args
        )
