from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from megatron.core import InferenceParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_tensor_model_parallel_group_if_none

from mbridge.models.qwen3_vl.transformer_config import Qwen3VLTransformerConfig


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
class Qwen3VLVisionPatchEmbed(nn.Module):

    def __init__(
        self,
        config: Qwen3VLTransformerConfig,
    ) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels,
                              self.embed_dim,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size,
                                           self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
class Qwen3VLVisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> torch.Tensor:
        if not hasattr(self, "inv_freq"):
            inv_freq = 1.0 / (self.theta
                              **(torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


@dataclass
class PatchMergerSubmodules:
    patch_norm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class Qwen3VLVisionPatchMerger(MegatronModule):

    def __init__(
        self,
        config: Qwen3VLTransformerConfig,
        submodules: PatchMergerSubmodules,
        use_postshuffle_norm=False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.input_size = config.hidden_size
        if self.use_postshuffle_norm:
            self.input_size = self.hidden_size
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=False)

        self.patch_norm = build_module(
            submodules.patch_norm,
            config=self.config,
            hidden_size=self.input_size,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc1",
            tp_group=tp_group,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            self.config.out_hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc1",
            tp_group=tp_group,
        )

    def forward(self, hidden_states):
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.patch_norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        output, _ = self.linear_fc2(hidden_states)

        return output


# only support to now
def split_deepstack_embs(
    visual_pos_masks: torch.Tensor,
    deepstack_visual_embeds: list[torch.Tensor],
    tp_size: int = 1,
    tp_rank: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
):
    # first split by cp (zigzag)
    assert cp_size == 1 and cp_rank == 0, "no support cp now"

    # split by tp
    if tp_size == 1 or visual_pos_masks is None:
        return visual_pos_masks, deepstack_visual_embeds

    assert visual_pos_masks.dim() == 2
    batch_size = visual_pos_masks.size(0)
    visual_pos_masks_list = visual_pos_masks.chunk(tp_size, dim=-1)
    embed_lens = [ele.sum(-1) for ele in visual_pos_masks_list]

    embed_lens = torch.stack(embed_lens, dim=-1)
    embed_cu_lens = embed_lens.view(-1).cumsum(dim=-1).tolist()
    embed_cu_lens = [0] + embed_cu_lens

    tp_slices = []
    for i in range(batch_size):
        idx = i * tp_size + tp_rank
        if embed_cu_lens[idx] != embed_cu_lens[idx + 1]:
            tp_slices.append(slice(embed_cu_lens[idx], embed_cu_lens[idx + 1]))

    deepstack_visual_embeds_ret = []
    for deepstack_visual_embed in deepstack_visual_embeds:
        tmp_slice_tensor = []
        for tp_slice in tp_slices:
            tmp_slice_tensor.append(deepstack_visual_embed[tp_slice])
        if len(tmp_slice_tensor) != 0:
            deepstack_visual_embeds_ret.append(torch.cat(tmp_slice_tensor, dim=0))
        else:
            deepstack_visual_embeds_ret.append(None)

    return visual_pos_masks_list[tp_rank], deepstack_visual_embeds_ret
