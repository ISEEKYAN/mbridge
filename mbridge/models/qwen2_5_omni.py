# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from transformers import AutoConfig

from ..core import VLMBridge, register_model


@register_model("qwen2_5_omni")
class Qwen2_5OmniBridge(VLMBridge):
    """
    Bridge implementation for Qwen 2.5-Omni models.

    Qwen 2.5-Omni has a nested config structure:
    - Qwen2_5OmniConfig -> thinker_config -> Qwen2_5OmniThinkerConfig -> text_config -> Qwen2_5OmniTextConfig
    """

    def _build_config(self):
        """
        Build the configuration for Qwen 2.5-Omni models.

        Returns:
            TransformerConfig: Configuration object for Qwen 2.5-Omni models
        """

        # Qwen2_5OmniConfig -> thinker_config -> text_config
        text_config = self.hf_config.thinker_config.text_config

        mrope_section = None
        if hasattr(text_config, "rope_parameters") and text_config.rope_parameters is not None:
            if isinstance(text_config.rope_parameters, dict):
                mrope_section = text_config.rope_parameters.get("mrope_section", None)
            elif hasattr(text_config.rope_parameters, "mrope_section"):
                mrope_section = text_config.rope_parameters.mrope_section

        # Temporarily replace hf_config with text_config so _build_base_config

        original_hf_config = self.hf_config
        self.hf_config = text_config

        try:
            config = self._build_base_config(
                add_bias_linear=False,
                # qwen specific
                add_qkv_bias=True,
                mrope_section=mrope_section,
            )
        finally:
            self.hf_config = original_hf_config

        return config