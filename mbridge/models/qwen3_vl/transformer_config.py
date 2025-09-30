from dataclasses import dataclass, field
from typing import List
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F

from megatron.core.transformer import TransformerConfig


@dataclass
class Qwen3VLTransformerConfig(TransformerConfig):
    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2304

    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])


def get_vision_model_config(config: Qwen3VLTransformerConfig, hf_config):
    config.num_moe_experts = None
    config.expert_model_parallel_size = 1
    config.moe_ffn_hidden_size = None

    config.num_layers = hf_config.depth
    config.ffn_hidden_size = hf_config.intermediate_size
    config.num_attention_heads = hf_config.num_heads  # num_heads
    config.add_bias_linear = True  # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True  # qkv_proj in attn has bias
    config.hidden_size = hf_config.hidden_size  # hidden_size
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.layernorm_epsilon = 1e-6
    config.apply_rotary_pos_emb_in_fp32 = True

    config.patch_size = hf_config.patch_size
    config.temporal_patch_size = hf_config.temporal_patch_size
    config.in_channels = hf_config.in_channels
    config.spatial_merge_size = hf_config.spatial_merge_size
    config.num_position_embeddings = hf_config.num_position_embeddings
    config.out_hidden_size = hf_config.out_hidden_size
    config.deepstack_visual_indexes = deepcopy(hf_config.deepstack_visual_indexes)

    config.gated_linear_unit = False  # no gated
    config.activation_func = partial(F.gelu, approximate="tanh")  # hidden_act
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads  # no GQA
    config.layernorm_zero_centered_gamma = False  # False
    config.apply_query_key_layer_scaling = False  # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False  # no swiglu, set false
    config.bias_dropout_fusion = False  # no dropout, set false
    config.attention_softmax_in_fp32 = True  # use True
    config.normalization = 'LayerNorm'

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.context_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None
    config.virtual_pipeline_model_parallel_size = 1
    config.pipeline_model_parallel_layout = None
    return config
