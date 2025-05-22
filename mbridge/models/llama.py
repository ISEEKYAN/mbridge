# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from ..core import LLMBridge, register_model
from ..core.bridge import Bridge, register_model


@register_model("llama")
class LLaMABridge(LLMBridge):
    """
    Bridge implementation for LLaMA2 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for LLaMA2 models, handling the conversion between
    Hugging Face LLaMA2 format and Megatron-Core.
    """

    def _build_config(self):
        """
        Build the configuration for LLaMA2 models.

        Configures LLaMA2-specific parameters such as attention bias settings.

        Returns:
            TransformerConfig: Configuration object for LLaMA2 models
        """
        qkv_bias = getattr(self.hf_config, "attention_bias", False)
        return self._build_base_config(
            use_cpu_initialization=False,
            add_bias_linear=False,
            add_qkv_bias=qkv_bias,
            **self.extra_args
        )

    def _get_gptmodel_args(self) -> dict:
        """
        Get GPT model arguments specific to LLaMA2.

        Handles LLaMA2-specific configurations such as RoPE scaling
        for extended context length models.

        Returns:
            dict: Dictionary of arguments for GPTModel initialization
        """
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                assert (
                    self.hf_config.rope_scaling["type"] == "linear"
                ), "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = (
                    self.hf_config.rope_scaling["factor"]
                )
        ret = dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )
        ret.update(rope_scaling_args)
        return ret
