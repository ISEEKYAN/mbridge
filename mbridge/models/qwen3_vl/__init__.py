from copy import deepcopy
from typing import Callable, Optional

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.extensions.transformer_engine import TENorm

from mbridge.core import register_model
from mbridge.core.util import unwrap_model
from mbridge.models.qwen3_vl.model import Qwen3VLModel
from mbridge.models.qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from mbridge.models.qwen3_vl.transformer_config import get_vision_model_config
from mbridge.models.qwen3_vl.utils import PatchMergerSubmodules
from mbridge.models.qwen3_vl.base_bridge import Qwen3VBaseBridge


_QWEN3VIT_DIRECT_MAPPING = {
    "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
    "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
    "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
    "vision_model.merger.patch_norm.weight": "model.visual.merger.norm.weight",
    "vision_model.merger.patch_norm.bias": "model.visual.merger.norm.bias",
    "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
}

_QWEN3VIT_ATTENTION_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
        "model.visual.blocks.{layer_number}.attn.proj.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.bias": [
        "model.visual.blocks.{layer_number}.attn.proj.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
        "model.visual.blocks.{layer_number}.attn.qkv.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
        "model.visual.blocks.{layer_number}.attn.qkv.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
        "model.visual.blocks.{layer_number}.norm1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_bias": [
        "model.visual.blocks.{layer_number}.norm1.bias",
    ],
}

_QWEN3VIT_MLP_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc1.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc2.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
        "model.visual.blocks.{layer_number}.norm2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_bias": [
        "model.visual.blocks.{layer_number}.norm2.bias",
    ],
}

_QWEN3VIT_OTHER_MAPPING = {
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.weight": [
        "model.visual.deepstack_merger_list.{layer_number}.norm.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.bias": [
        "model.visual.deepstack_merger_list.{layer_number}.norm.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.weight": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc1.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.bias": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc1.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.weight": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc2.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.bias": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc2.bias",
    ],
}


@register_model("qwen3_vl")
class Qwen3VLBridge(Qwen3VBaseBridge):
    """
    Bridge implementation for Qwen3VL models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen3VL models, handling the conversion between
    Hugging Face Qwen3VL format and Megatron-Core.
    """

    TransformerConfigClass = Qwen3VLTransformerConfig

    def _build_config(self):
        """
        Build the configuration for LLaMA2 models.

        Configures LLaMA2-specific parameters such as attention bias settings.

        Returns:
            TransformerConfig: Configuration object for LLaMA2 models
        """
        assert False, "dense qwen3_vl support comming soon"
        return self._build_base_config(
            # qwen specific
            text_config_key="text_config",
            qk_layernorm=True,
            mrope_section=self.hf_config.text_config.rope_scaling.get("mrope_section",
                                                                      [24, 20, 20]),
        )


@register_model("qwen3_vl_moe")
class Qwen3VLMoEBridge(Qwen3VBaseBridge):

    TransformerConfigClass = Qwen3VLTransformerConfig
    _DIRECT_MAPPING = {
        **_QWEN3VIT_DIRECT_MAPPING,
        "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        **_QWEN3VIT_ATTENTION_MAPPING,
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
    }

    _MLP_MAPPING = {
        **_QWEN3VIT_MLP_MAPPING,
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
    }

    _OTHER_MAPPING = {
        **_QWEN3VIT_OTHER_MAPPING,
    }

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config.text_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["language_model.output_layer.weight"] = (
                "model.embed_tokens.weight")

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config.text_config, "tie_word_embeddings", False):
            return ["model.embed_tokens.weight"]
        return []

    def _set_extra_config(self):
        self.configpatch_size = self.hf_config.vision_config.patch_size
        self.configtemporal_patch_size = self.hf_config.vision_config.temporal_patch_size
        self.configin_channels = self.hf_config.vision_config.in_channels
        self.configspatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        self.confignum_position_embeddings = self.hf_config.vision_config.num_position_embeddings
        self.configout_hidden_size = self.hf_config.vision_config.out_hidden_size
        self.configdeepstack_visual_indexes = deepcopy(
            self.hf_config.vision_config.deepstack_visual_indexes)

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        if name.startswith("vision_model.") or \
            ".pre_mlp_layernorm.weight" in name or \
            ".mlp.router.weight" in name:
            return super()._weight_name_mapping_mlp(name)

        assert ".mlp.experts.linear_fc" in name
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        key = key.split(".weight")[0] + ".weight"
        convert_names = []
        mapping_names = self._MLP_MAPPING[key]
        convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _model_provider(self, post_model_creation_callbacks: list[Callable[[torch.nn.Module],
                                                                           None]]):
        """
        Creates and returns a model provider function.

        The returned function creates a GPTModel with the specified configuration
        when called with pre_process and post_process parameters.

        Args:
            post_model_creation_callbacks: List of callbacks to be called after model creation

        Returns:
            function: A provider function that creates and returns a GPTModel instance
        """

        share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        def provider(pre_process,
                     post_process,
                     add_decoder=True,
                     add_encoder=True,
                     vp_stage: Optional[int] = None):
            self._set_extra_config()
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            vision_transformer_config = get_vision_model_config(deepcopy(self.config),
                                                                self.hf_config.vision_config)
            vision_transformer_config.pipeline_model_parallel_size = 1
            vision_transformer_config.first_pipeline_num_layers = None

            vision_patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

            setattr(self, "vision_config", vision_transformer_config)

            model = Qwen3VLModel(
                language_transformer_config=self.config,
                language_transformer_layer_spec=transformer_layer_spec,
                language_vocab_size=self.hf_config.text_config.vocab_size,
                language_max_sequence_length=self.hf_config.text_config.max_position_embeddings,
                vision_transformer_config=vision_transformer_config,
                vision_transformer_layer_spec=vision_transformer_layer_spec,
                vision_patch_merger_spec=vision_patch_merger_spec,
                language_rotary_base=self.hf_config.text_config.rope_theta,
                pre_process=pre_process,
                post_process=post_process,
                add_decoder=add_decoder,
                add_encoder=add_encoder,
                parallel_output=True,
                language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            )

            for callback in post_model_creation_callbacks:
                callback(
                    model,
                    pre_process=pre_process,
                    post_process=post_process,
                    config=self.config,
                    hf_config=self.hf_config,
                )

            return model

        return provider

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
            async_tensor_model_parallel_allreduce=True,
            variable_seq_lengths=False,
            batch_p2p_comm=True,
            distribute_saved_activations=False,
            cp_comm_type='p2p',
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            mrope_section=self.hf_config.text_config.rope_scaling.get(
                "mrope_section",
                [24, 20, 20],
            ),
        )
