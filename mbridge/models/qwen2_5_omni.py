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

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "thinker.model.norm.weight",
        "output_layer.weight": "thinker.lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "thinker.model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "thinker.model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "thinker.model.layers.{layer_number}.self_attn.q_proj.weight",
            "thinker.model.layers.{layer_number}.self_attn.k_proj.weight",
            "thinker.model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "thinker.model.layers.{layer_number}.self_attn.q_proj.bias",
            "thinker.model.layers.{layer_number}.self_attn.k_proj.bias",
            "thinker.model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "thinker.model.layers.{layer_number}.mlp.gate_proj.weight",
            "thinker.model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "thinker.model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["thinker.model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.
        
        Override to access vocab_size, max_position_embeddings, and rope_theta
        """
        text_config = self.hf_config.thinker_config.text_config
        
        return dict(
            vocab_size=text_config.vocab_size,
            max_sequence_length=text_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=getattr(text_config, "rope_theta", None) or 
                        (text_config.rope_parameters.rope_theta 
                        if hasattr(text_config, "rope_parameters") and 
                            text_config.rope_parameters is not None 
                        else 1000000.0),
        )


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