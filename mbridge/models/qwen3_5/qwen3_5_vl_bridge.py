import torch

from mbridge.core import register_model
from mbridge.models.qwen3_5.transformer_config import Qwen3p5VLTransformerConfig
from mbridge.models.qwen3_5.base_bridge import Qwen3p5VlBaseBridge


_QWEN3p5VIT_DIRECT_MAPPING = {
    "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
    "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
    "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
    "vision_model.merger.norm.weight": "model.visual.merger.norm.weight",
    "vision_model.merger.norm.bias": "model.visual.merger.norm.bias",
    "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
}

_QWEN3p5_VISUAL_MAPPING = {
    # visual attn
    "vision_model.blocks.{layer_number}.attn.proj.weight": [
        "model.visual.blocks.{layer_number}.attn.proj.weight",
    ],
    "vision_model.blocks.{layer_number}.attn.proj.bias": [
        "model.visual.blocks.{layer_number}.attn.proj.bias",
    ],
    "vision_model.blocks.{layer_number}.attn.qkv.bias": [
        "model.visual.blocks.{layer_number}.attn.qkv.bias",
    ],
    "vision_model.blocks.{layer_number}.attn.qkv.weight": [
        "model.visual.blocks.{layer_number}.attn.qkv.weight",
    ],
    # visual mlp
    "vision_model.blocks.{layer_number}.mlp.linear_fc1.weight": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc1.weight",
    ],
    "vision_model.blocks.{layer_number}.mlp.linear_fc1.bias": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc1.bias",
    ],
    "vision_model.blocks.{layer_number}.mlp.linear_fc2.weight": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc2.weight",
    ],
    "vision_model.blocks.{layer_number}.mlp.linear_fc2.bias": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc2.bias",
    ],
    # visual norm
    "vision_model.blocks.{layer_number}.norm1.weight": [
        "model.visual.blocks.{layer_number}.norm1.weight",
    ],
    "vision_model.blocks.{layer_number}.norm1.bias": [
        "model.visual.blocks.{layer_number}.norm1.bias",
    ],
    "vision_model.blocks.{layer_number}.norm2.weight": [
        "model.visual.blocks.{layer_number}.norm2.weight",
    ],
    "vision_model.blocks.{layer_number}.norm2.bias": [
        "model.visual.blocks.{layer_number}.norm2.bias",
    ],
}

_QWEN3p5TEXT_DIRECT_MAPPING = {
    "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
    "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
    "language_model.output_layer.weight": "lm_head.weight",
}

_QWEN3p5TEXT_ATTENTION_MAPPING = {
    "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
        "model.language_model.layers.{layer_number}.self_attn.o_proj.weight",
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
        "model.language_model.layers.{layer_number}.input_layernorm.weight",
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
        "model.language_model.layers.{layer_number}.self_attn.q_norm.weight",
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
        "model.language_model.layers.{layer_number}.self_attn.k_norm.weight",
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
        "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
        "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
        "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
        "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
        "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
        "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
    ],
    # linear attetnion
    "language_model.decoder.layers.{layer_number}.self_attention.dt_bias": [
        "model.language_model.layers.{layer_number}.linear_attn.dt_bias"
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.A_log": [
        "model.language_model.layers.{layer_number}.linear_attn.A_log"
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.in_proj.weight": [
        "model.language_model.layers.{layer_number}.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.{layer_number}.linear_attn.in_proj_z.weight",
        "model.language_model.layers.{layer_number}.linear_attn.in_proj_b.weight",
        "model.language_model.layers.{layer_number}.linear_attn.in_proj_a.weight"
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.conv1d.weight": [
        "model.language_model.layers.{layer_number}.linear_attn.conv1d.weight"
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.out_norm.weight": [
        "model.language_model.layers.{layer_number}.linear_attn.norm.weight"
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.out_proj.weight": [
        "model.language_model.layers.{layer_number}.linear_attn.out_proj.weight"
    ],
    "language_model.decoder.layers.{layer_number}.self_attention.in_proj.layer_norm_weight": [
        "model.language_model.layers.{layer_number}.input_layernorm.weight"    
    ]
}

_QWEN3p5TEXT_MLP_MAPPING = {
    "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
        "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
        "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
    ],
    "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
        "model.language_model.layers.{layer_number}.post_attention_layernorm.weight",
    ],
    "language_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
        "model.language_model.layers.{layer_number}.mlp.down_proj.weight",
    ],
}

_QWEN3p5TEXT_MOE_MLP_MAPPING = {
    "language_model.decoder.layers.{layer_number}.pre_mlp_layernorm.weight": [
        "model.language_model.layers.{layer_number}.post_attention_layernorm.weight",
    ],
    "language_model.decoder.layers.{layer_number}.mlp.router.weight": [
        "model.language_model.layers.{layer_number}.mlp.gate.weight",
    ],
    "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc1.weight": [
        "model.language_model.layers.{layer_number}.mlp.experts.gate_up_proj",
    ],
    "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc2.weight": [
        "model.language_model.layers.{layer_number}.mlp.experts.down_proj",
    ],
    # shared expert
    "language_model.decoder.layers.{layer_number}.mlp.shared_experts.linear_fc1.weight": [
        "model.language_model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
    ],
    "language_model.decoder.layers.{layer_number}.mlp.shared_experts.linear_fc2.weight": [
        "model.language_model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
    ],
    "language_model.decoder.layers.{layer_number}.mlp.shared_experts.gate_weight": [
        "model.language_model.layers.{layer_number}.mlp.shared_expert_gate.weight"
    ],
}

@register_model("qwen3_5")
class Qwen3p5VlBridge(Qwen3p5VlBaseBridge):
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    except:
        Qwen3_5VisionModel = None
    HfVisionClass: type = Qwen3_5VisionModel
    TransformerConfigClass = Qwen3p5VLTransformerConfig

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": "attention_dropout",
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    _DIRECT_MAPPING = {
        **_QWEN3p5VIT_DIRECT_MAPPING,
        **_QWEN3p5TEXT_DIRECT_MAPPING,
    }
    _ATTENTION_MAPPING = {
        **_QWEN3p5TEXT_ATTENTION_MAPPING,
    }
    _MLP_MAPPING = {
        **_QWEN3p5TEXT_MLP_MAPPING,
    }
    _OTHER_MAPPING = {}
    _VISUAL_MAPPING = {
        **_QWEN3p5_VISUAL_MAPPING,
    }

    def _build_config(self):
        return self._build_base_config(
            text_config_key="text_config",
            layernorm_epsilon=self.hf_config.text_config.rms_norm_eps,
            use_cpu_initialization=False,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
            async_tensor_model_parallel_allreduce=True,
            distribute_saved_activations=False,
            cp_comm_type="p2p",
            # Qwen3.5 specific
            qk_layernorm=True,
            layernorm_zero_centered_gamma=True,
            attention_output_gate=True,
            kv_channels=self.hf_config.text_config.head_dim,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=self.hf_config.text_config.full_attention_interval,
            linear_conv_kernel_dim=self.hf_config.text_config.linear_conv_kernel_dim,
            linear_key_head_dim=self.hf_config.text_config.linear_key_head_dim,
            linear_value_head_dim=self.hf_config.text_config.linear_value_head_dim,
            linear_num_key_heads=self.hf_config.text_config.linear_num_key_heads,
            linear_num_value_heads=self.hf_config.text_config.linear_num_value_heads,
            rotary_percent=self.hf_config.text_config.rope_scaling.get("partial_rotary_factor", 0.25),
            rotary_interleaved=self.hf_config.text_config.rope_scaling.get("mrope_interleaved", True),
            mrope_section=self.hf_config.text_config.rope_scaling.get(
                "mrope_section",
                [11, 11, 10],
            ),
            patch_size=self.hf_config.vision_config.patch_size,
            temporal_patch_size=self.hf_config.vision_config.temporal_patch_size,
            in_channels=self.hf_config.vision_config.in_channels,
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            num_position_embeddings=self.hf_config.vision_config.num_position_embeddings,
            out_hidden_size=self.hf_config.vision_config.out_hidden_size,
            apply_rotary_pos_emb_in_fp32=True,
        )


@register_model("qwen3_5_moe")
class Qwen3p5MoeVlBridge(Qwen3p5VlBaseBridge):
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
    except:
        Qwen3_5MoeVisionModel = None
    HfVisionClass: type = Qwen3_5MoeVisionModel

    TransformerConfigClass = Qwen3p5VLTransformerConfig

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "moe_intermediate_size",
        "attention_dropout": "attention_dropout",
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    _DIRECT_MAPPING = {
        **_QWEN3p5VIT_DIRECT_MAPPING,
        **_QWEN3p5TEXT_DIRECT_MAPPING,
    }
    _ATTENTION_MAPPING = {
        **_QWEN3p5TEXT_ATTENTION_MAPPING,
    }
    _MLP_MAPPING = {
        **_QWEN3p5TEXT_MOE_MLP_MAPPING,
    }
    _OTHER_MAPPING = {}
    _VISUAL_MAPPING = {
        **_QWEN3p5_VISUAL_MAPPING,
    }

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        if (
            name.startswith("vision_model.")
            or ".pre_mlp_layernorm.weight" in name
            or ".mlp.router.weight" in name
            or ".shared_experts" in name
        ):
            return super()._weight_name_mapping_mlp(name)

        assert ".mlp.experts.linear_fc" in name, f"{name=}"
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        key = key.split(".weight")[0] + ".weight"
        convert_names = []
        mapping_names = self._MLP_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _build_config(self):
        return self._build_base_config(
            text_config_key="text_config",
            layernorm_epsilon=self.hf_config.text_config.rms_norm_eps,
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.text_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.text_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.text_config.num_experts,
            moe_aux_loss_coeff=self.hf_config.text_config.router_aux_loss_coef,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_shared_expert_overlap=True,
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            moe_shared_expert_intermediate_size=self.hf_config.text_config.shared_expert_intermediate_size,
            moe_shared_expert_gate=self.hf_config.text_config.shared_expert_intermediate_size > 0,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
            async_tensor_model_parallel_allreduce=True,
            distribute_saved_activations=False,
            cp_comm_type="p2p",
            # Qwen3.5 specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            layernorm_zero_centered_gamma=True,
            attention_output_gate=True,
            kv_channels=self.hf_config.text_config.head_dim,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=self.hf_config.text_config.full_attention_interval,
            linear_conv_kernel_dim=self.hf_config.text_config.linear_conv_kernel_dim,
            linear_key_head_dim=self.hf_config.text_config.linear_key_head_dim,
            linear_value_head_dim=self.hf_config.text_config.linear_value_head_dim,
            linear_num_key_heads=self.hf_config.text_config.linear_num_key_heads,
            linear_num_value_heads=self.hf_config.text_config.linear_num_value_heads,
            rotary_percent=self.hf_config.text_config.rope_scaling.get("partial_rotary_factor", 0.25),
            rotary_interleaved=self.hf_config.text_config.rope_scaling.get("mrope_interleaved", True),
            mrope_section=self.hf_config.text_config.rope_scaling.get(
                "mrope_section",
                [11, 11, 10],
            ),
            patch_size=self.hf_config.vision_config.patch_size,
            temporal_patch_size=self.hf_config.vision_config.temporal_patch_size,
            in_channels=self.hf_config.vision_config.in_channels,
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            num_position_embeddings=self.hf_config.vision_config.num_position_embeddings,
            out_hidden_size=self.hf_config.vision_config.out_hidden_size,
            apply_rotary_pos_emb_in_fp32=True,
        )
