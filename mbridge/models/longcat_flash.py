# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import inspect
from typing import Callable, Optional

import torch

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnBackend
from .longcat.gpt_layer_specs import get_shortcut_decoder_block_spec
from .longcat.transformer_config import MLATransformerConfig

from ..core import LLMBridge, register_model


@register_model("longcat_flash")
class LongCatFlashBridge(LLMBridge):
    """
    Specific bridge implementation for LongCatFlash models.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _MLP_MAPPING = {
        "mlp.0.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.0.weight"
        ],
        "mlp.1.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.1.weight"
        ],

        "mlp.0.linear_fc2.weight": ["model.layers.{layer_number}.mlps.0.down_proj.weight"],
        "mlp.1.linear_fc2.weight": ["model.layers.{layer_number}.mlps.1.down_proj.weight"],

        "mlp.0.linear_fc1.weight": [
            "model.layers.{layer_number}.mlps.0.gate_proj.weight",
            "model.layers.{layer_number}.mlps.0.up_proj.weight",
        ],
        "mlp.1.linear_fc1.weight": [
            "model.layers.{layer_number}.mlps.1.gate_proj.weight",
            "model.layers.{layer_number}.mlps.1.up_proj.weight",
        ],

        "pre_mlp_layernorm.0.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.0.weight"
        ],
        "pre_mlp_layernorm.1.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.1.weight"
        ],

        "moe.router.weight": ["model.layers.{layer_number}.mlp.router.classifier.weight"],
        "moe.router.expert_bias": [
            "model.layers.{layer_number}.mlp.router.e_score_correction_bias"
        ],

        "moe.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "moe.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }

    _ATTENTION_MAPPING = {
        "input_layernorm.0.weight": [
            "model.layers.{layer_number}.input_layernorm.0.weight"
        ],
        "input_layernorm.1.weight": [
            "model.layers.{layer_number}.input_layernorm.1.weight"
        ],

        "self_attention.0.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.0.o_proj.weight"
        ],
        "self_attention.1.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.1.o_proj.weight"
        ],

        "self_attention.0.linear_q_proj.weight": [
            "model.layers.{layer_number}.self_attn.0.q_proj.weight"
        ],
        "self_attention.1.linear_q_proj.weight": [
            "model.layers.{layer_number}.self_attn.1.q_proj.weight"
        ],

        # NOTE: linear_qkv_down_proj must appear BEFORE linear_kv_down_proj because
        # "kv_down_proj" is a substring of "qkv_down_proj" and the mapping uses
        # substring matching (keyword in name).
        "self_attention.0.linear_qkv_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.0.q_a_proj.weight",
            "model.layers.{layer_number}.self_attn.0.kv_a_proj_with_mqa.weight"
        ],
        "self_attention.1.linear_qkv_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.1.q_a_proj.weight",
            "model.layers.{layer_number}.self_attn.1.kv_a_proj_with_mqa.weight"
        ],

        "self_attention.0.linear_kv_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.0.kv_a_proj_with_mqa.weight"
        ],
        "self_attention.1.linear_kv_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.1.kv_a_proj_with_mqa.weight"
        ],

        "self_attention.0.kv_layernorm": [
            "model.layers.{layer_number}.self_attn.0.kv_a_layernorm.weight"
        ],
        "self_attention.1.kv_layernorm": [
            "model.layers.{layer_number}.self_attn.1.kv_a_layernorm.weight"
        ],

        "self_attention.0.linear_kv_up_proj.weight": [
            "model.layers.{layer_number}.self_attn.0.kv_b_proj.weight"
        ],
        "self_attention.1.linear_kv_up_proj.weight": [
            "model.layers.{layer_number}.self_attn.1.kv_b_proj.weight"
        ],

        "self_attention.0.linear_q_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.0.q_a_proj.weight"
        ],
        "self_attention.1.linear_q_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.1.q_a_proj.weight"
        ],

        "self_attention.0.linear_q_up_proj.weight": [
            "model.layers.{layer_number}.self_attn.0.q_b_proj.weight"
        ],
        "self_attention.1.linear_q_up_proj.weight": [
            "model.layers.{layer_number}.self_attn.1.q_b_proj.weight"
        ],

        "self_attention.0.linear_q_up_proj.layer_norm_weight": [
            "model.layers.{layer_number}.self_attn.0.q_a_layernorm.weight"
        ],
        "self_attention.1.linear_q_up_proj.layer_norm_weight": [
            "model.layers.{layer_number}.self_attn.1.q_a_layernorm.weight"
        ],
    }

    _SHARED_STATE_DICT_MAPPING = {
        "embedding.word_embeddings.weight": [
            "model.embed_tokens.weight",
        ],
        "output_layer.weight": [
            "lm_head.weight",
        ],
    }

    TransformerConfigClass = MLATransformerConfig

    def _build_config(self):
        # from verl.models.mcore.patch_v012 import apply_patch
        # apply_patch()

        hf_config = self.hf_config
        mla_rope_config = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "rope",
        }
        if "rope_scaling" in hf_config and hf_config.rope_scaling is not None:
            mla_rope_config.update(hf_config.rope_scaling)

        if not hasattr(hf_config, "num_hidden_layers"):
            hf_config.num_hidden_layers = hf_config.num_layers
        moe_layer_freq = [1] * hf_config.num_hidden_layers

        if not hasattr(hf_config, "intermediate_size"):
            hf_config.intermediate_size = hf_config.ffn_hidden_size
        if not hasattr(hf_config, "moe_intermediate_size"):
            hf_config.moe_intermediate_size = hf_config.expert_ffn_hidden_size

        base_config = {
            "attention_backend": AttnBackend.fused,
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "ffn_hidden_size": hf_config.intermediate_size,
            "qk_layernorm": True,
            # moe specific
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_bias_update_rate": 0.001,
            "moe_router_enable_expert_bias": True,
            "moe_router_topk": hf_config.moe_topk,
            "num_moe_experts": hf_config.n_routed_experts,
            "zero_expert_num": hf_config.zero_expert_num,
            "zero_expert_type": hf_config.zero_expert_type,
            # "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size * hf_config.n_shared_experts,
            "moe_aux_loss_coeff": getattr(hf_config, "aux_loss_alpha", 0.001),
            # moe_router_load_balancing_type="seq_aux_loss",
            "moe_router_load_balancing_type": "none",  # default None for RL
            "moe_shared_expert_overlap": True,
            "moe_grouped_gemm": True,
            "moe_router_score_function": "softmax",
            "moe_router_pre_softmax": True,
            "moe_router_topk_scaling_factor": hf_config.routed_scaling_factor,
            "moe_layer_freq": moe_layer_freq,
            "moe_permute_fusion": True,
            # MLA
            "q_lora_rank": hf_config.q_lora_rank,
            "kv_lora_rank": hf_config.kv_lora_rank,
            "qk_head_dim": hf_config.qk_nope_head_dim,
            "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
            "v_head_dim": hf_config.v_head_dim,
            "rotary_base": hf_config.rope_theta,
            "rotary_scaling_factor": mla_rope_config["factor"],
            "rope_type": mla_rope_config["type"],
            "mscale": mla_rope_config["mscale"],
            "mscale_all_dim": mla_rope_config["mscale_all_dim"],
            "beta_fast": mla_rope_config["beta_fast"],
            "beta_slow": mla_rope_config["beta_slow"],
            # mcore 0.12 moe
            "moe_router_dtype": "fp32",
            "disable_bf16_reduced_precision_matmul": True,
            # other
            "persist_layer_norm": True,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
        }
        
        import megatron.core
        megatron_version = getattr(megatron.core, '__version__')
        if megatron_version >= "0.14":
            base_config["original_max_position_embeddings"] = mla_rope_config["original_max_position_embeddings"]
        else:
            base_config["max_position_embeddings"] = mla_rope_config["original_max_position_embeddings"]

        return self._build_base_config(**base_config)

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.

        Constructs a dictionary of arguments required to initialize a GPTModel
        based on the configuration.

        Returns:
            dict: A dictionary of arguments for GPTModel initialization
        """
        ret = dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )

        return ret

    def _get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        """
        Gets the transformer layer specification.

        Creates and returns a specification for the transformer layers based on
        the current configuration.

        Returns:
            TransformerLayerSpec: Specification for transformer layers

        Raises:
            AssertionError: If normalization is not RMSNorm
        """
        # check if get_shortcut_decoder_block_spec has vp_stage parameter
        sig = inspect.signature(get_shortcut_decoder_block_spec)
        self.has_vp_stage = "vp_stage" in sig.parameters  # for mcore 0.12 compatibility
        extra_args = {}
        if self.has_vp_stage:
            extra_args["vp_stage"] = vp_stage
        transformer_layer_spec = get_shortcut_decoder_block_spec(
            self.config, use_transformer_engine=True, **extra_args
        )
        return transformer_layer_spec

    def get_model(self, *args, **kwargs):
        model = super().get_model(*args, **kwargs)
        # Maintain router bias dtype for LongCat's moe structure (l.moe.router)
        from mbridge.core.util import unwrap_model
        for m in model:
            m = unwrap_model(m)
            if hasattr(m, "decoder"):
                for l in m.decoder.layers:
                    if (
                        hasattr(l, "moe")
                        and hasattr(l.moe, "router")
                        and hasattr(l.moe.router, "_maintain_float32_expert_bias")
                    ):
                        l.moe.router._maintain_float32_expert_bias()
        return model

    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        """
        Creates and returns a model provider function.

        The returned function creates a GPTModel with the specified configuration
        when called with pre_process and post_process parameters.

        Args:
            post_model_creation_callbacks: List of callbacks to be called after model creation

        Returns:
            function: A provider function that creates and returns a GPTModel instance
        """

        share_embeddings_and_output_weights = getattr(
            self.hf_config, "tie_word_embeddings", False
        )

        def provider(pre_process, post_process, vp_stage: Optional[int] = None):
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            gptmodel_args = self._get_gptmodel_args()
            if vp_stage is not None and self.has_vp_stage:
                gptmodel_args["vp_stage"] = vp_stage

            model = GPTModel(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                **gptmodel_args,
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

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if (
            "self_attention" in mcore_weights_name
            or "input_layernorm" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name or "moe" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        # Merge q_latent and kv_latent for MLA (before super() to avoid dtype issues)
        if (
            "self_attention." in mcore_weights_name
            and "linear_qkv_down_proj." in mcore_weights_name
        ):
            if hasattr(self, "dtype") and self.dtype is not None:
                hf_weights = [
                    w.to(self.dtype) if w.dtype != self.dtype else w for w in hf_weights
                ]
            assert len(hf_weights) == 2
            q_latent, kv_latent = hf_weights
            qkv_latent = torch.cat([q_latent, kv_latent], dim=0).view(-1, self.hf_config.hidden_size).contiguous()
            return qkv_latent

        # Keep router weights in original dtype (e.g. float32) to preserve precision
        if "router" in mcore_weights_name:
            if len(hf_weights) == 1:
                return hf_weights[0]
            raise NotImplementedError(f"Unsupported router parameter: {mcore_weights_name}")

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:

        # Handle QKV down proj
        if "linear_qkv_down_proj." in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            # split latent
            assert len(hf_names) == 2
            qkv_latent = mcore_weights.view(-1, self.hf_config.hidden_size)
            q_latent = qkv_latent[:self.hf_config.q_lora_rank, :]
            kv_latent = qkv_latent[self.hf_config.q_lora_rank:, :]
            return hf_names, [q_latent, kv_latent]

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [
                            x.format(layer_number=layer_number, expert_id=expert_id)
                            for x in mapping_names
                        ]
                    )
                else:
                    convert_names.extend(
                        [x.format(layer_number=layer_number) for x in mapping_names]
                    )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names
