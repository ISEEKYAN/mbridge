# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from ..core import LLMBridge, register_model


@register_model("qwen2_moe")
class Qwen2MoEBridge(LLMBridge):
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }
    _MLP_MAPPING = {
        "shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        ],
        "pre_mlp_layernorm": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "shared_experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
        ],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
        "shared_experts.gate_weight": [
            "model.layers.{layer_number}.mlp.shared_expert_gate.weight"
        ],
        "mlp.experts.linear_fc1": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }

    # transformers>=5: fused expert tensors [num_experts, ...]
    _MLP_MAPPING_MOE_FUSED = {
        "decoder.layers.{layer_number}.mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj",
        ],
        "decoder.layers.{layer_number}.mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.down_proj",
        ],
    }

    def _build_config(self):
        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_experts,
            moe_shared_expert_intermediate_size=self.hf_config.shared_expert_intermediate_size,
            moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            # moe_router_load_balancing_type="aux_loss",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_shared_expert_overlap=True,
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen specific
            moe_router_pre_softmax=True,
            add_qkv_bias=True,
        )

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        if (
            self._hf_moe_stacked_layout()
            and "mlp.experts.linear_fc" in name
        ):
            split_name = name.split(".")
            split_name[2] = "{layer_number}"
            key = ".".join(split_name)
            pre, _expert = key.split(".weight", 1)
            stacked_key = pre + ".weight"
            mapping_names = self._MLP_MAPPING_MOE_FUSED[stacked_key]
            return [x.format(layer_number=layer_number) for x in mapping_names]

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

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        if (
            self._hf_moe_stacked_layout()
            and "mlp.experts.linear_fc" in mcore_weights_name
            and len(hf_names) == 1
        ):
            experts_key = hf_names[0]
            experts_idx = int(mcore_weights_name.split(".weight")[-1])
            if experts_key not in self.export_weights_buff:
                self.export_weights_buff[experts_key] = {}
            assert experts_idx not in self.export_weights_buff[experts_key]
            self.export_weights_buff[experts_key][experts_idx] = mcore_weights
            if (
                len(self.export_weights_buff[experts_key])
                < self.config.num_moe_experts
            ):
                return [], []
            ordered = [
                self.export_weights_buff[experts_key].pop(i)
                for i in range(self.config.num_moe_experts)
            ]
            self.export_weights_buff.pop(experts_key)
            return [experts_key], [torch.stack(ordered)]
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        if (
            hasattr(self, "dtype")
            and self.dtype is not None
            and "expert_bias" not in mcore_weights_name
        ):
            hf_weights = [
                w.to(self.dtype) if w.dtype != self.dtype else w for w in hf_weights
            ]
        if (
            self._hf_moe_stacked_layout()
            and "mlp.experts.linear_fc" in mcore_weights_name
            and len(hf_weights) == 1
        ):
            local_experts_idx = int(mcore_weights_name.split(".weight")[-1])
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            experts_idx = (
                local_experts_idx + num_experts_per_rank * self.mpu.ep_rank
            )
            return hf_weights[0][experts_idx].clone().contiguous()
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
