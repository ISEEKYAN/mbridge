# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Callable, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec, get_gpt_decoder_block_spec

from ..core import register_model
from .qwen2 import Qwen2Bridge


@register_model("mimo")
class MimoBridge(Qwen2Bridge):
    """
    Bridge implementation for Mimo models.

    This class extends Qwen2Bridge to provide specific configurations and
    optimizations for Mimo models, handling the conversion between
    Hugging Face Mimo format and Megatron-Core.

    MiMo adds MTP (Multi-Token Prediction) layers on top of Qwen2 architecture.
    """

    def _build_config(self):
        """Override to add MTP configuration."""
        hf_config = self.hf_config

        # Add MTP configuration if present
        mtp_args = {}
        if "num_nextn_predict_layers" in hf_config:
            mtp_args["mtp_num_layers"] = hf_config.num_nextn_predict_layers
            mtp_args["mtp_loss_scaling_factor"] = self.extra_args.get("mtp_loss_scaling_factor", 0.1)

        return self._build_base_config(
            add_qkv_bias=True,
            qk_layernorm=False,
            **mtp_args,
        )

    def _get_gptmodel_args(self) -> dict:
        """Override to add MTP block spec if needed."""
        ret = super()._get_gptmodel_args()

        # Add MTP block spec if MTP layers are present
        if self.config.mtp_num_layers is not None:
            transformer_layer_spec = get_gpt_decoder_block_spec(config=self.config,
                                                                use_transformer_engine=True)
            mtp_block_spec = get_gpt_mtp_block_spec(self.config,
                                                    transformer_layer_spec.layer_specs[-1],
                                                    True)
            ret["mtp_block_spec"] = mtp_block_spec

        return ret

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """Override to handle MTP layer mappings."""
        # Check if this is an MTP layer weight
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)

        # Otherwise use parent class mapping
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _convert_mtp_param(self, name: str) -> list[str]:
        """Convert MTP layer parameters from MCore to HF format."""
        # For now, assume single MTP layer support
        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        # Get the MTP layer index
        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}

        # Direct mappings for MTP-specific components
        direct_name_mapping = {
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": f"model.mtp_layers.{mtp_layer_idx}.token_layernorm.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": f"model.mtp_layers.{mtp_layer_idx}.hidden_layernorm.weight",
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": f"model.mtp_layers.{mtp_layer_idx}.input_proj.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": f"model.mtp_layers.{mtp_layer_idx}.final_layernorm.weight",
        }

        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        # Handle transformer components within MTP
        # Check if this is a transformer_layer component
        if "transformer_layer" in name:
            # Create a proxy name to use with parent class methods
            # Convert mtp.layers.{idx}.transformer_layer.* to decoder.layers.{idx}.*
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name:
                convert_names = super()._weight_name_mapping_mlp(proxy_name)
            else:
                raise NotImplementedError(
                    f"Unsupported transformer component in MTP: {name}"
                )

            # Replace the layer index in converted names to point to mtp_layers
            convert_names = [
                cn.replace(
                    f"model.layers.{mtp_layer_idx}", f"model.mtp_layers.{mtp_layer_idx}"
                )
                for cn in convert_names
            ]
            return convert_names
        else:
            raise NotImplementedError(f"Unsupported MTP parameter name: {name}")
        return convert_names


    # The default MTP input concatenation order in Megatron is [target_token_embed, hidden_state],
    # which is applicable to DeepSeek and GLM models;
    # The MTP input format of the Mimo-7B-RL model is [hidden_state, target_token_embed],
    # with the order completely reversed;
    # It is necessary to swap the parameter values of the first half and the second half of eh_proj here.
    #
    # mimo mtp sglang code:
    #   https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/mimo_mtp.py#L61-L69
    # mimo mtp vllm code:
    #   https://github.com/XiaomiMiMo/vllm/blob/feat_mimo_mtp_stable_073/vllm/model_executor/models/mimo_mtp.py#L84-L85
    # megatron code:
    #   https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/multi_token_prediction.py#L636

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        """Swap halves of eh_proj weights before handing off to Megatron-Core."""
        weight = super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
        print(mcore_weights_name)
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = weight.chunk(2, dim=1)
            weight = torch.cat([second_half, first_half], dim=1)
        return weight

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Swap halves back when exporting eh_proj weights to HuggingFace format."""
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = mcore_weights.chunk(2, dim=1)
            mcore_weights = torch.cat([second_half, first_half], dim=1)
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
