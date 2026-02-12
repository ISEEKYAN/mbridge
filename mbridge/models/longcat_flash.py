# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import inspect
from typing import Callable, Optional

import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
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

    _MTP_ATTENTION_MAPPING = {
        # input_layernorm - directly under mtp.layers, not in transformer_layer
        "mtp.layers.{mtp_layer}.transformer_layer.input_layernorm.weight": [
            "model.mtp.layers.{mtp_layer}.input_layernorm.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_qkv_down_proj.weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.q_a_proj.weight",
            "model.mtp.layers.{mtp_layer}.self_attn.kv_a_proj_with_mqa.weight",
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_down_proj.weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.q_a_proj.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_up_proj.layer_norm_weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.q_a_layernorm.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_up_proj.weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.q_b_proj.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_kv_down_proj.weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.kv_a_proj_with_mqa.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.kv_layernorm.weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.kv_a_layernorm.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_kv_up_proj.weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.kv_b_proj.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_proj.weight": [
            "model.mtp.layers.{mtp_layer}.self_attn.o_proj.weight"
        ],
    }

    _MTP_MLP_MAPPING = {
        "mtp.layers.{mtp_layer}.transformer_layer.mlp.linear_fc1.weight": [
            "model.mtp.layers.{mtp_layer}.transformer_layer.mlp.gate_proj.weight",
            "model.mtp.layers.{mtp_layer}.transformer_layer.mlp.up_proj.weight",
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.mlp.linear_fc2.weight": [
            "model.mtp.layers.{mtp_layer}.transformer_layer.mlp.down_proj.weight"
        ],
        "mtp.layers.{mtp_layer}.transformer_layer.pre_mlp_layernorm.weight": [
            "model.mtp.layers.{mtp_layer}.post_attention_layernorm.weight"
        ],
    }

    _MTP_EXTRA_MAPPING = {
        "mtp.layers.{mtp_layer}.eh_proj.weight": [
            "model.mtp.layers.{mtp_layer}.eh_proj.weight"
        ],
        "mtp.layers.{mtp_layer}.enorm.weight": [
            "model.mtp.layers.{mtp_layer}.enorm.m.weight"
        ],
        "mtp.layers.{mtp_layer}.hnorm.weight": [
            "model.mtp.layers.{mtp_layer}.hnorm.m.weight"
        ],
    }

    _MTP_GLOBAL_NORM_MAPPING = {
        "mtp.layers.0.final_layernorm.weight": [
            "model.mtp.norm.weight"
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

        mtp_args = {}
        if "num_nextn_predict_layers" in hf_config:
            mtp_args["mtp_num_layers"] = hf_config.num_nextn_predict_layers
            mtp_args["mtp_loss_scaling_factor"] = 0.1

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

        base_config.update(mtp_args)
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

        if self.config.mtp_num_layers is not None and self.config.mtp_num_layers > 0:
            from megatron.core.models.gpt.gpt_layer_specs import get_mtp_transformer_layer_with_transformer_engine_spec

            mtp_transformer_layer_spec = get_mtp_transformer_layer_with_transformer_engine_spec(
                qk_layernorm=True,
            )
            mtp_block_spec = get_gpt_mtp_block_spec(
                self.config, mtp_transformer_layer_spec, use_transformer_engine=True
            )
            ret["mtp_block_spec"] = mtp_block_spec

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

    def _freeze_mtp_weights(self, model: torch.nn.Module):
        """
        Freeze MTP-specific weights to prevent training updates.

        Args:
            model: The GPTModel instance
        """
        if hasattr(model, 'mtp') and model.mtp is not None:
            # Freeze all MTP parameters
            for param in model.mtp.parameters():
                param.requires_grad = False
            frozen_count = sum(p.numel() for p in model.mtp.parameters() if not p.requires_grad)
            print(f"[MTP] Frozen MTP-specific module weights ({frozen_count} parameters)")
        if (
            hasattr(model, 'mtp_embedding')
            and model.mtp_embedding is not None
            and not model.pre_process  # Only freeze if mtp_embedding is independent
        ):
            for param in model.mtp_embedding.parameters():
                param.requires_grad = False
            frozen_count = sum(p.numel() for p in model.mtp_embedding.parameters() if not p.requires_grad)
            print(f"[MTP] Frozen MTP embedding module weights ({frozen_count} parameters)")
        elif hasattr(model, 'mtp_embedding') and model.mtp_embedding is not None and model.pre_process:
            # mtp_embedding is shared with embedding (pre_process=True), don't freeze
            print(f"[MTP] MTP embedding is shared with main embedding (pre_process=True), keeping it trainable")
            

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

            # Freeze MTP weights if MTP layers are present
            if self.config.mtp_num_layers is not None and self.config.mtp_num_layers > 0:
                self._freeze_mtp_weights(model)

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

        # Handle mtp_embedding weights
        # When pre_process=True and mtp_process=True, mtp_embedding is the same as embedding
        # so they will be handled by the normal embedding mapping.
        # When pre_process=False and mtp_process=True, mtp_embedding is independent and frozen,
        # so it uses the same HF weights as the main embedding (model.embed_tokens.weight).
        # After loading, mtp_embedding is frozen and won't be updated during training.
        if "mtp_embedding" in mcore_weights_name:
            # mtp_embedding uses the same HF weights as main embedding
            if "word_embeddings.weight" in mcore_weights_name:
                return ["model.embed_tokens.weight"]
            else:
                # Other mtp_embedding weights not covered above
                return []

        # Skip MTP block weights - they are handled by _convert_mtp_param
        if "mtp" in mcore_weights_name and "mtp_embedding" not in mcore_weights_name:
            # This is a non-embedding MTP weight, convert it
            return self._convert_mtp_param(mcore_weights_name)

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

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:

        # Handle mtp_embedding weights - they are tied to main model embedding in checkpoint
        # so we should NOT save them separately. They will be loaded from the same HF keys
        # as the main embedding (model.embed_tokens.weight, etc.) but kept frozen during training.
        # Skipping mtp_embedding here prevents overwriting the main embedding weights during save.
        if "mtp_embedding" in mcore_weights_name:
            # Skip all mtp_embedding weights during export
            # They are tied to main embedding in sharded_state_dict and will be restored from there
            return [], []

        # Handle QKV down proj
        if "linear_qkv_down_proj." in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            # split latent
            assert len(hf_names) == 2
            qkv_latent = mcore_weights.view(-1, self.hf_config.hidden_size)
            q_latent = qkv_latent[:self.hf_config.q_lora_rank, :]
            kv_latent = qkv_latent[self.hf_config.q_lora_rank:, :]
            return hf_names, [q_latent, kv_latent]

        # Handle shared weights between MTP and main model
        if (
            self.config.mtp_num_layers is not None
            and self.config.mtp_num_layers > 0
            and mcore_weights_name in self._SHARED_STATE_DICT_MAPPING
        ):
            hf_names = self._SHARED_STATE_DICT_MAPPING[mcore_weights_name]
            return hf_names, [mcore_weights] * len(hf_names)

        # Handle MTP weights export (non-embedding weights)
        if (
            self.config.mtp_num_layers is not None
            and self.config.mtp_num_layers > 0
            and "mtp" in mcore_weights_name
        ):
            try:
                hf_names = self._weight_name_mapping_mcore_to_hf(
                    mcore_weights_name
                )
                # If hf_names is empty, it means this is a shared embedding
                # weight that should be skipped
                if not hf_names:
                    return [], []

                # Special handling for MTP MLP weights that need splitting
                # MTP uses the same linear_fc1 merging as decoder layers:
                # - Megatron: mtp.layers.{idx}.transformer_layer.mlp.linear_fc1.weight (merged)
                # - HuggingFace: model.mtp.layers.{idx}.transformer_layer.mlp.gate_proj.weight + up_proj.weight (split)
                if (
                    "linear_fc1.weight" in mcore_weights_name
                    or "linear_fc1.bias" in mcore_weights_name
                ):
                    # Split gate_proj and up_proj (same as decoder layer MLP)
                    assert len(hf_names) == 2, f"Expected 2 HF names for MTP linear_fc1, got {len(hf_names)}: {hf_names}"
                    gate, up = mcore_weights.chunk(2)
                    return hf_names, [gate, up]

                # For other MTP weights, no splitting needed
                return hf_names, [mcore_weights] * len(hf_names)
            except NotImplementedError:
                # Skip unsupported MTP weights
                print(
                    f"[MTP] Skipping unsupported MTP weight: "
                    f"{mcore_weights_name}"
                )
                return [], []

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

    def _convert_mtp_param(self, name: str) -> list[str]:
        # Check if this is the global norm (not per-layer)
        if "final_layernorm" in name:
            # This is mtp.final_layernorm.weight -> model.mtp.norm.weight
            for mcore_template, hf_names in self._MTP_GLOBAL_NORM_MAPPING.items():
                if mcore_template in name:
                    return hf_names
            raise NotImplementedError(f"Invalid MTP global norm parameter name: {name}")

        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "mtp" or parts[1] != "layers":
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        mtp_layer_idx = int(parts[2])

        template_name = name.replace(f".{mtp_layer_idx}.", ".{mtp_layer}.")

        all_mappings = [
            self._MTP_EXTRA_MAPPING,
            self._MTP_ATTENTION_MAPPING,
            self._MTP_MLP_MAPPING,
        ]

        for mapping in all_mappings:
            for mcore_template, hf_names in mapping.items():
                if template_name == mcore_template:
                    return [
                        hf_name.format(mtp_layer=mtp_layer_idx)
                        for hf_name in hf_names
                    ]

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")
