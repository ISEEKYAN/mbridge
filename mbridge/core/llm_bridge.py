# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Generator

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

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

    def export_weights(
        self, models: list[torch.nn.Module]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:

        # map vpp layer number to global layer number
        def get_layer_number(vpp_rank: int, local_layer_number: int) -> int:
            unwrapped_model = unwrap_model(models[vpp_rank])
            global_layer_number = (
                unwrapped_model.decoder.layers[local_layer_number].layer_number - 1
            )
            return global_layer_number

        def get_model_chunk_generator():
            for model in models:
                yield from model.named_parameters()

        weights_names = []
        for vpp_rank, model in enumerate(models):
            for name, param in model.named_parameters():
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))
        weights_names_all_pp = [None] * self.mpu.pp_size
        torch.distributed.all_gather_object(
            object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group
        )
        weights_names_all_pp = sum(weights_names_all_pp, [])
        model_chunk_generator = get_model_chunk_generator()
        for iter_pp_rank, iter_vpp_rank, iter_name in weights_names_all_pp:
            if iter_pp_rank == self.mpu.pp_rank:
                try:
                    name, param = next(model_chunk_generator)
                except StopIteration:
                    name, param = None, None
                if "layers" in iter_name:
                    local_layer_number = int(
                        iter_name.split("layers.")[1].split(".")[0]
                    )
                    global_layer_number = get_layer_number(
                        iter_vpp_rank, local_layer_number
                    )
                    name = iter_name.replace(
                        f"layers.{local_layer_number}.",
                        f"layers.{global_layer_number}.",
                    )
            else:
                name, param = None, None

            name = broadcast_str_from_megatron_pp(name)
            broad_pp_param = broadcast_from_megatron_pp(param)

            while name.startswith("module."):
                name = name[len("module.") :]

            # EP
            if ".mlp.experts.linear_fc" in name and self.mpu.ep_size > 1:
                num_experts = self.config.num_moe_experts
                num_experts_per_rank = num_experts // self.mpu.ep_size
                infer_params = [
                    torch.empty_like(broad_pp_param) for _ in range(self.mpu.ep_size)
                ]
                torch.distributed.all_gather(
                    infer_params, broad_pp_param, group=self.mpu.ep_group
                )

                name_prefix, local_expert_id = name.split(".weight")
                local_expert_id = int(local_expert_id)
                global_expert_ids = [
                    num_experts_per_rank * ep_rank + local_expert_id
                    for ep_rank in range(self.mpu.ep_size)
                ]
                global_expert_names = [
                    f"{name_prefix}.weight{expert_id}"
                    for expert_id in global_expert_ids
                ]

                for name, param in zip(global_expert_names, infer_params):
                    if self.mpu.etp_size > 1:
                        # gather etp
                        etp_params = [
                            torch.empty_like(param) for _ in range(self.mpu.etp_size)
                        ]
                        torch.distributed.all_gather(
                            etp_params, param, group=self.mpu.etp_group
                        )
                        params = etp_params
                    else:
                        params = [param]

                    merge_params = self._weight_merge_across_tp(
                        name, params, broad_pp_param
                    )
                    if not isinstance(merge_params, list):
                        merge_params = [merge_params]
                    converted_names, converted_params = self._weight_to_hf_format(
                        name, merge_params
                    )

                    yield from zip(converted_names, converted_params)
                continue

            # TP
            if (
                hasattr(broad_pp_param, "tensor_model_parallel")
                and broad_pp_param.tensor_model_parallel
            ):
                # allocate a new tensor with proper size
                if self.mpu.tp_size <= 1:
                    infer_params = [broad_pp_param]
                else:
                    infer_params = [
                        torch.empty_like(broad_pp_param)
                        for _ in range(self.mpu.tp_size)
                    ]
                    torch.distributed.all_gather(
                        infer_params, broad_pp_param, group=self.mpu.tp_group
                    )
                infer_params = self._weight_merge_across_tp(
                    name, infer_params, broad_pp_param
                )
            else:
                infer_params = broad_pp_param

            converted_names, converted_params = self._weight_to_hf_format(
                name, infer_params
            )

            yield from zip(converted_names, converted_params)
