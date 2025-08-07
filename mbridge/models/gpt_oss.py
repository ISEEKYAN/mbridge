# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Callable,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.dot_product_attention import (
    DotProductAttention as MCoreDotProductAttention,
)
from megatron.core.transformer.enums import AttnBackend

from ..core import LLMBridge, register_model
from ..core.util import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    get_model,
    unwrap_model,
)


@dataclass
class GPTOSSConfig(TransformerConfig):
    """
    Base config for GPT-OSS
    """

    hidden_size: int = 2880
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 2880
    kv_channels: Optional[int] = 64
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = True
    share_embeddings_and_output_weights: bool = False
    vocab_size: int = 201088
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    position_embedding_type: str = "yarn"
    rotary_base: int = 150000
    rotary_scaling_factor: float = 32.0
    yarn_original_max_position_embeddings: int = 131072
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_correction_range_round_to_int: bool = False

    moe_router_topk: int = 4
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_ffn_hidden_size: int = 2880
    moe_router_load_balancing_type: str = "none"
    seq_length: int = 131072
    window_size: Optional[Tuple[int, int]] = (128, 0)
    softmax_type: Literal["vanilla", "off-by-one", "learnable"] = "learnable"
    activation_func: Callable = quick_gelu
    glu_linear_offset: float = 1.0
    bias_activation_fusion: bool = True
    window_attn_skip_freq: Optional[Union[int, List[int]]] = 2  # alternative SWA/full
    attention_backend: AttnBackend = AttnBackend.local  # supports "local" and "fused"


@register_model("gpt_oss")
class GPTOSSBridge(LLMBridge):
    """
    Specific bridge implementation for GPT-OSS models.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _MLP_MAPPING = {
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_proj.bias": [
            "model.layers.{layer_number}.self_attn.o_proj.bias"
        ],
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj",
        ],
        "mlp.experts.linear_fc1.bias": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj_bias",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.down_proj",
        ],
        "mlp.experts.linear_fc2.bias": [
            "model.layers.{layer_number}.mlp.experts.down_proj_bias",
        ],
    }

    _ATTENTION_MAPPING = {
        "self_attention.core_attention.softmax_offset": [
            "model.layers.{layer_number}.self_attn.sinks",
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
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_proj.bias": [
            "model.layers.{layer_number}.self_attn.o_proj.bias"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
    }

    TransformerConfigClass = GPTOSSConfig

    def _build_config(self):
        hf_config = self.hf_config
        dtype = self.dtype
        overlap_p2p_comm = self.mpu.vpp_size is not None and self.mpu.pp_size > 1
        batch_p2p_comm = False
        base_config = {
            # Model architecture parameters
            "num_moe_experts": hf_config.num_local_experts,
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_query_groups": hf_config.num_key_value_heads,
            "ffn_hidden_size": hf_config.intermediate_size,
            "attention_dropout": hf_config.attention_dropout,
            "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
            "kv_channels": getattr(hf_config, "head_dim", None),
            "layernorm_epsilon": hf_config.rms_norm_eps,
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
            "use_cpu_initialization": False,
            "overlap_p2p_comm": overlap_p2p_comm,
            "batch_p2p_comm": batch_p2p_comm,
        }
        base_config.update(self.extra_args)

        cfg = GPTOSSConfig(**base_config)
        print(cfg)
        return cfg

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
            position_embedding_type=self.config.position_embedding_type,
            rotary_base=self.hf_config.rope_theta,
        )

        return ret

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

            if (
                self.config.attention_backend == AttnBackend.local
                and self.config.softmax_type != "vanilla"
            ):
                for layer in transformer_layer_spec.layer_specs:
                    if hasattr(layer, "submodules"):
                        layer.submodules.self_attention.submodules.core_attention = (
                            MCoreDotProductAttention
                        )
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
            or "input_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _get_safetensor_io(self, weights_path: str):
        if self.dtype == torch.bfloat16:
            from .ext.gpt_oss.dequant_mxfp4_safetensor_io import (
                DequantMXFP4SafeTensorIO,
            )

            return DequantMXFP4SafeTensorIO(self._get_actual_hf_path(weights_path))
        else:
            raise NotImplemented("only support bfloat16 for now")

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "experts.linear_fc" in name:
                    weight_or_bias = "weight" if "weight" in name else "bias"
                    expert_id = name.split(weight_or_bias)[-1]
                    convert_names.extend(
                        [
                            (x.format(layer_number=layer_number), int(expert_id))
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

    def export_weights(
        self, models: list[torch.nn.Module]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        models = [unwrap_model(model) for model in models]

        def get_model_chunk_generator():
            for model in models:
                existing_keys = set()
                for name, param in model.named_parameters():
                    existing_keys.add(name)
                    yield name, param

                # note
                # there is a bug in megatron GPTModel
                # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
                # for now we patch it by adding those keys to extra_keys.
                extra_keys = [
                    x
                    for x in model.state_dict().keys()
                    if "_extra_state" not in x
                    and "expert_bias" in x
                    and x not in existing_keys
                ]
                for name in extra_keys:
                    yield name, model.state_dict()[name].to(torch.cuda.current_device())

        weights_names = []
        for vpp_rank, model in enumerate(models):
            existing_keys = set()
            for name, param in model.named_parameters():
                existing_keys.add(name)
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))
            extra_keys = [
                x
                for x in model.state_dict().keys()
                if "_extra_state" not in x
                and "expert_bias" in x
                and x not in existing_keys
            ]
            for name in extra_keys:
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))

        weights_names_all_pp = [None] * self.mpu.pp_size
        torch.distributed.all_gather_object(
            object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group
        )
        weights_names_all_pp = sum(weights_names_all_pp, [])
        model_chunk_generator = get_model_chunk_generator()
        local_to_global_maps = [
            self._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)
            for model in models
        ]
        for iter_pp_rank, iter_vpp_rank, iter_name in weights_names_all_pp:
            local_to_global_map = local_to_global_maps[iter_vpp_rank]
            if iter_pp_rank == self.mpu.pp_rank:
                try:
                    name, param = next(model_chunk_generator)
                except StopIteration:
                    name, param = None, None
                name = local_to_global_map[iter_name]
            else:
                name, param = None, None

            name = broadcast_str_from_megatron_pp(name)
            broad_pp_param = broadcast_from_megatron_pp(param)

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
                weight_or_bias = "weight" if "weight" in name else "bias"
                name_prefix, local_expert_id = name.split(f".{weight_or_bias}")
                local_expert_id = int(local_expert_id)
                global_expert_ids = [
                    num_experts_per_rank * ep_rank + local_expert_id
                    for ep_rank in range(self.mpu.ep_size)
                ]
                global_expert_names = [
                    f"{name_prefix}.{weight_or_bias}{expert_id}"
                    for expert_id in global_expert_ids
                ]
                all_experts = {}
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

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Import Hugging Face weights to MCore format.

        Takes Hugging Face weight names and tensors, outputs MCore weight tensor.
        Due to MCore's runtime optimizations involving weight merging, input is a list.

        Args:
            mcore_weights_name: MCore weight name
            hf_weights: List of Hugging Face weight tensors

        Returns:
            torch.Tensor: MCore weight tensor

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            head_dim = getattr(
                self.hf_config, "head_dim", hidden_dim // num_attention_heads
            )
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // group_dim
            q = q.view(
                [
                    real_num_key_value_heads,
                    group_dim,
                    -1,
                ]
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qkv
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            # merge gate_proj and up_proj
            assert len(hf_weights) == 1

            def interleave(elem):
                return torch.cat((elem[::2, ...], elem[1::2, ...]), dim=0)

            return interleave(hf_weights[0])
        if len(hf_weights) == 1:
            return hf_weights[0]
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
