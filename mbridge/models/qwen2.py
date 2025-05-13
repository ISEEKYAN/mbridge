# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from ..core import LLMBridge, register_model


@register_model("qwen2")
class Qwen2Bridge(LLMBridge):
    """
    Bridge implementation for Qwen2 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen2 models, handling the conversion between
    Hugging Face Qwen2 format and Megatron-Core.
    """

    def _build_config(self):
        """
        Build the configuration for Qwen2 models.

        Configures Qwen2-specific parameters such as QKV bias settings and
        layer normalization options.

        Returns:
            TransformerConfig: Configuration object for Qwen2 models
        """
        return self._build_base_config(
            use_cpu_initialization=False,
            add_bias_linear=False,
            # qwen2
            add_qkv_bias=True,
            qk_layernorm=False,
            **self.extra_args
        )
