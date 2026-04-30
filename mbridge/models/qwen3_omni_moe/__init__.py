from copy import deepcopy

from mbridge.core import register_model
from mbridge.models.qwen3_omni_moe.base_bridge import Qwen3OmniBaseBridge
from mbridge.models.qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from mbridge.utils.hf_config import get_hf_rope_scaling


_QWEN3AUDIO_DIRECT_MAPPING = {
    "audio_model.ln_post.weight":"thinker.audio_tower.ln_post.weight",
    "audio_model.ln_post.bias":"thinker.audio_tower.ln_post.bias",
    "audio_model.conv2d1.weight":"thinker.audio_tower.conv2d1.weight",
    "audio_model.conv2d1.bias":"thinker.audio_tower.conv2d1.bias",
    "audio_model.conv2d2.weight":"thinker.audio_tower.conv2d2.weight",
    "audio_model.conv2d2.bias":"thinker.audio_tower.conv2d2.bias",
    "audio_model.conv2d3.weight":"thinker.audio_tower.conv2d3.weight",
    "audio_model.conv2d3.bias":"thinker.audio_tower.conv2d3.bias",
    "audio_model.conv_out.weight":"thinker.audio_tower.conv_out.weight",
    "audio_model.proj1.weight":"thinker.audio_tower.proj1.weight",
    "audio_model.proj1.bias":"thinker.audio_tower.proj1.bias",
    "audio_model.proj2.weight":"thinker.audio_tower.proj2.weight",
    "audio_model.proj2.bias":"thinker.audio_tower.proj2.bias",
}

_QWEN3AUDIO_ATTENTION_MAPPING = {
    "audio_model.layers.{layer_number}.self_attn.k_proj.weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.k_proj.weight",
    ],
    "audio_model.layers.{layer_number}.self_attn.k_proj.bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.k_proj.bias",
    ],
    "audio_model.layers.{layer_number}.self_attn.v_proj.weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.v_proj.weight",
    ],
    "audio_model.layers.{layer_number}.self_attn.v_proj.bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.v_proj.bias",
    ],
    "audio_model.layers.{layer_number}.self_attn.q_proj.weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.q_proj.weight",
    ],
    "audio_model.layers.{layer_number}.self_attn.q_proj.bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.q_proj.bias",
    ],
    "audio_model.layers.{layer_number}.self_attn.out_proj.weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.out_proj.weight",
    ],
    "audio_model.layers.{layer_number}.self_attn.out_proj.bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.out_proj.bias",
    ],
    "audio_model.layers.{layer_number}.self_attn_layer_norm.weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn_layer_norm.weight",
    ],
    "audio_model.layers.{layer_number}.self_attn_layer_norm.bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn_layer_norm.bias",
    ],
    "audio_model.layers.{layer_number}.fc1.weight": [
        "thinker.audio_tower.layers.{layer_number}.fc1.weight",
    ],
    "audio_model.layers.{layer_number}.fc1.bias": [
        "thinker.audio_tower.layers.{layer_number}.fc1.bias",
    ],
    "audio_model.layers.{layer_number}.fc2.weight": [
        "thinker.audio_tower.layers.{layer_number}.fc2.weight",
    ],
    "audio_model.layers.{layer_number}.fc2.bias": [
        "thinker.audio_tower.layers.{layer_number}.fc2.bias",
    ],
    "audio_model.layers.{layer_number}.final_layer_norm.weight": [
        "thinker.audio_tower.layers.{layer_number}.final_layer_norm.weight",
    ],
    "audio_model.layers.{layer_number}.final_layer_norm.bias": [
        "thinker.audio_tower.layers.{layer_number}.final_layer_norm.bias",
    ],
}

_QWEN3VIT_DIRECT_MAPPING = {
    "vision_model.patch_embed.proj.weight": "thinker.visual.patch_embed.proj.weight",
    "vision_model.patch_embed.proj.bias": "thinker.visual.patch_embed.proj.bias",
    "vision_model.pos_embed.weight": "thinker.visual.pos_embed.weight",
    "vision_model.merger.patch_norm.weight": "thinker.visual.merger.ln_q.weight",
    "vision_model.merger.patch_norm.bias": "thinker.visual.merger.ln_q.bias",
    "vision_model.merger.linear_fc1.weight": "thinker.visual.merger.mlp.0.weight",
    "vision_model.merger.linear_fc1.bias": "thinker.visual.merger.mlp.0.bias",
    "vision_model.merger.linear_fc2.weight": "thinker.visual.merger.mlp.2.weight",
    "vision_model.merger.linear_fc2.bias": "thinker.visual.merger.mlp.2.bias",
}

_QWEN3VIT_ATTENTION_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
        "thinker.visual.blocks.{layer_number}.attn.proj.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.bias": [
        "thinker.visual.blocks.{layer_number}.attn.proj.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
        "thinker.visual.blocks.{layer_number}.attn.qkv.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
        "thinker.visual.blocks.{layer_number}.attn.qkv.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
        "thinker.visual.blocks.{layer_number}.norm1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_bias": [
        "thinker.visual.blocks.{layer_number}.norm1.bias",
    ],
}

_QWEN3VIT_MLP_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc1.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc2.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
        "thinker.visual.blocks.{layer_number}.norm2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_bias": [
        "thinker.visual.blocks.{layer_number}.norm2.bias",
    ],
}

_QWEN3VIT_OTHER_MAPPING = {
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.weight": [
        "thinker.visual.merger_list.{layer_number}.ln_q.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.bias": [
        "thinker.visual.merger_list.{layer_number}.ln_q.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.weight": [
        "thinker.visual.merger_list.{layer_number}.mlp.0.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.bias": [
        "thinker.visual.merger_list.{layer_number}.mlp.0.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.weight": [
        "thinker.visual.merger_list.{layer_number}.mlp.2.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.bias": [
        "thinker.visual.merger_list.{layer_number}.mlp.2.bias",
    ],
}


@register_model("qwen3_omni_moe")
class Qwen3OmniMoeBridge(Qwen3OmniBaseBridge):
    """
    Bridge implementation for Qwen3 Omni Moe models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen3 Omni Moe models, handling the conversion between
    Hugging Face Qwen3 Omni Moe format and Megatron-Core.
    """

    TransformerConfigClass = Qwen3VLTransformerConfig
    
    _DIRECT_MAPPING = {
        **_QWEN3AUDIO_DIRECT_MAPPING,
        **_QWEN3VIT_DIRECT_MAPPING,
        "language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
        "language_model.output_layer.weight": "thinker.lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        **_QWEN3VIT_ATTENTION_MAPPING,
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "thinker.model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "thinker.model.layers.{layer_number}.input_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
            "thinker.model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
            "thinker.model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "thinker.model.layers.{layer_number}.self_attn.q_proj.weight",
            "thinker.model.layers.{layer_number}.self_attn.k_proj.weight",
            "thinker.model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "thinker.model.layers.{layer_number}.self_attn.q_proj.bias",
            "thinker.model.layers.{layer_number}.self_attn.k_proj.bias",
            "thinker.model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    _MLP_MAPPING = {
        **_QWEN3VIT_MLP_MAPPING,
        "language_model.decoder.layers.{layer_number}.pre_mlp_layernorm.weight": [
            "thinker.model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.router.weight": [
            "thinker.model.layers.{layer_number}.mlp.gate.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc1.weight{expert_number}": [
            "thinker.model.layers.{layer_number}.mlp.experts.{expert_number}.gate_proj.weight",
            "thinker.model.layers.{layer_number}.mlp.experts.{expert_number}.up_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc2.weight{expert_number}": [
            "thinker.model.layers.{layer_number}.mlp.experts.{expert_number}.down_proj.weight",
        ],
    }

    _OTHER_MAPPING = {
        **_QWEN3AUDIO_ATTENTION_MAPPING,
        **_QWEN3VIT_OTHER_MAPPING,
    }

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        if (
            name.startswith("vision_model.")
            or ".pre_mlp_layernorm.weight" in name
            or ".mlp.router.weight" in name
        ):
            return super()._weight_name_mapping_mlp(name)

        assert ".mlp.experts.linear_fc" in name
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        split_name = key.split(".weight")
        expert_number = split_name[1]
        key = split_name[0] + ".weight{expert_number}"
        convert_names = []
        mapping_names = self._MLP_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number, expert_number=expert_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _build_config(self):
        hf_config_backup = self.hf_config
        self.hf_config = self.hf_config.thinker_config
        base_config = self._build_base_config(
            text_config_key="text_config",
            layernorm_epsilon=self.hf_config.text_config.rms_norm_eps,
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.text_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.text_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.text_config.num_experts,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_dtype="fp32",
            # moe_router_load_balancing_type="aux_loss",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
            distribute_saved_activations=False,
            cp_comm_type="p2p",
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            # qwen3vl specific
            mrope_section=get_hf_rope_scaling(self.hf_config.text_config).get(
                "mrope_section",
                [24, 20, 20],
            ),
            patch_size=self.hf_config.vision_config.patch_size,
            temporal_patch_size=self.hf_config.vision_config.temporal_patch_size,
            in_channels=self.hf_config.vision_config.in_channels,
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            num_position_embeddings=self.hf_config.vision_config.num_position_embeddings,
            out_hidden_size=self.hf_config.vision_config.out_hidden_size,
            deepstack_visual_indexes=deepcopy(
                self.hf_config.vision_config.deepstack_visual_indexes
            ),
        )

        self.hf_config = hf_config_backup
        
        return base_config
