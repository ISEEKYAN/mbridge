# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Generator

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from torch.nn import functional as F

from .bridge import Bridge
from .layer import LinearForLastLayer
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
        }

        # Update with any provided overrides
        base_config.update(kwargs)
        # print(f"Overridden TF init config: {base_config}")

        return TransformerConfig(**base_config)

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

    def _get_transformer_layer_spec(self):
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
        transformer_layer_spec = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True
        )
        return transformer_layer_spec

    def _model_provider(
        self, share_embeddings_and_output_weights=False, value_model=False
    ):
        """
        Creates and returns a model provider function.

        The returned function creates a GPTModel with the specified configuration
        when called with pre_process and post_process parameters.

        Args:
            share_embeddings_and_output_weights: Whether to share embedding weights
            value_model: Whether this is a value model with a custom output layer

        Returns:
            function: A provider function that creates and returns a GPTModel instance
        """

        def provider(pre_process, post_process):
            transformer_layer_spec = self._get_transformer_layer_spec()
            gptmodel_args = self._get_gptmodel_args()
            model = GPTModel(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                **gptmodel_args,
            )
            if post_process and value_model:
                model.output_layer = LinearForLastLayer(
                    input_size=self.config.hidden_size,
                    output_size=1,
                    config=self.config,
                )

            return model

        return provider
