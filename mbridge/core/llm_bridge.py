# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import inspect
from typing import Callable, Generator, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from torch.nn import functional as F

from .bridge import Bridge
from .util import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    unwrap_model,
)


class LLMBridge(Bridge):
    """
    Bridge implementation for Large Language Models.

    This class extends the base Bridge class to provide specific functionality
    for handling Large Language Models (LLMs) like GPT models.
    """

    TransformerConfigClass = TransformerConfig

    def _build_base_config(self, **kwargs):
        """
        Build the base configuration for the model.

        Args:
            **kwargs: Additional configuration overrides

        Returns:
            TransformerConfig: The constructed transformer configuration
        """
        hf_config = self.hf_config
        dtype = self.dtype
        overlap_p2p_comm = self.mpu.vpp_size is not None and self.mpu.pp_size > 1
        batch_p2p_comm = False
        base_config = {
            # Model architecture parameters
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_query_groups": hf_config.num_key_value_heads,
            "ffn_hidden_size": hf_config.intermediate_size,
            "attention_dropout": hf_config.attention_dropout,
            "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
            "kv_channels": getattr(hf_config, "head_dim", None),
            "layernorm_epsilon": hf_config.rms_norm_eps,
            # Activation and normalization
            "activation_func": F.silu,
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
            # Data types
            "pipeline_dtype": dtype,
            "params_dtype": dtype,
            "bf16": dtype is torch.bfloat16,
            # Parallel configuration
            "tensor_model_parallel_size": self.mpu.tp_size,
            "pipeline_model_parallel_size": self.mpu.pp_size,
            "expert_model_parallel_size": self.mpu.ep_size,
            "expert_tensor_parallel_size": self.mpu.etp_size,
            "virtual_pipeline_model_parallel_size": self.mpu.vpp_size,
            "context_parallel_size": self.mpu.cp_size,
            "sequence_parallel": self.mpu.tp_size > 1,
            # Common settings
            "variable_seq_lengths": True,
            "masked_softmax_fusion": True,
            "moe_token_dispatcher_type": "alltoall",
            "add_bias_linear": False,
            "use_cpu_initialization": False,
            "overlap_p2p_comm": overlap_p2p_comm,
            "batch_p2p_comm": batch_p2p_comm,
        }

        # Update with any provided overrides
        base_config.update(kwargs)
        base_config.update(self.extra_args)

        return self.TransformerConfigClass(**base_config)

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.

        Constructs a dictionary of arguments required to initialize a GPTModel
        based on the configuration.

        Returns:
            dict: A dictionary of arguments for GPTModel initialization
        """
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )

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
        assert (
            self.config.normalization == "RMSNorm"
        ), "only RMSNorm is supported for now"
        # check if get_gpt_decoder_block_spec has vp_stage parameter
        sig = inspect.signature(get_gpt_decoder_block_spec)
        self.has_vp_stage = "vp_stage" in sig.parameters  # for mcore 0.12 compatibility
        extra_args = {}
        if self.has_vp_stage:
            extra_args["vp_stage"] = vp_stage
        transformer_layer_spec = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True, **extra_args
        )
        return transformer_layer_spec

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
