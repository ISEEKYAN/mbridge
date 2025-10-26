from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from torch import nn
from torch.nn import functional as F

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
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
class Qwen3VLVisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> torch.Tensor:
        if not hasattr(self, "inv_freq"):
            inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0,
                        self.dim,
                        2,
                        dtype=torch.float,
                        device=torch.cuda.current_device(),
                    )
                    / self.dim
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
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


def split_part_by_cp_tp(cp_size, cp_rank, tp_size, tp_rank, split_size):
    part_list = list(range(split_size))

    cp_rank2 = 2 * cp_size - cp_rank - 1
    cp_part_list = part_list[cp_rank * tp_size:(cp_rank + 1) *
                             tp_size] + part_list[cp_rank2 * tp_size:(cp_rank2 + 1) * tp_size]

    assert len(cp_part_list) % tp_size == 0
    echo_tp_len = len(cp_part_list) // tp_size
    cp_tp_part_list = cp_part_list[tp_rank * echo_tp_len:(tp_rank + 1) * echo_tp_len]
    return cp_tp_part_list


def split_deepstack_embs(
    visual_pos_masks: torch.Tensor,
    deepstack_visual_embeds: list[torch.Tensor],
    tp_size: int = 1,
    tp_rank: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
):
    split_size = tp_size
    if cp_size > 1:
        split_size *= (cp_size * 2)
    if split_size == 1 or visual_pos_masks is None:
        return visual_pos_masks, deepstack_visual_embeds

    assert visual_pos_masks.dim() == 2
    assert visual_pos_masks.shape[-1] % split_size == 0
    batch_size = visual_pos_masks.size(0)

    # first split by cp(zigzag), then split by sp
    # for example cp=2/tp=4
    # visual_pos_masks will split in 16 part:
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # first split by cp(zigzag) is:
    # cp_rank0: [0, 1, 2, 3, 12, 13, 14, 15]
    # cp_rank1: [4, 5, 6, 7, 8, 9, 10, 11]
    # then split by sp:
    # cp_rank0/tp_rank0 = [0, 1]
    # cp_rank0/tp_rank1 = [2, 3]
    # ...
    # cp_rank1/tp_rank2 = [8, 9]
    # cp_rank1/tp_rank3 = [10, 11]
    cp_tp_part_list = split_part_by_cp_tp(cp_size, cp_rank, tp_size, tp_rank, split_size)
    visual_pos_masks_list = visual_pos_masks.chunk(split_size, dim=-1)
    embed_lens = [ele.sum(-1) for ele in visual_pos_masks_list]

    embed_lens = torch.stack(embed_lens, dim=-1)
    embed_cu_lens = embed_lens.view(-1).cumsum(dim=-1).tolist()
    assert len(embed_cu_lens) == split_size * batch_size
    embed_cu_lens = [0] + embed_cu_lens

    cp_tp_slices = []
    for i in range(batch_size):
        for idx in cp_tp_part_list:
            idx += i * split_size
            cp_tp_slices.append(slice(embed_cu_lens[idx], embed_cu_lens[idx + 1]))

    deepstack_visual_embeds_ret = []
    for deepstack_visual_embed in deepstack_visual_embeds:
        tmp_slice_tensor = []
        for tp_slice in cp_tp_slices:
            tmp_slice_tensor.append(deepstack_visual_embed[tp_slice])
        deepstack_visual_embeds_ret.append(torch.cat(tmp_slice_tensor, dim=0))

    visual_pos_masks_ret = torch.cat([visual_pos_masks_list[i] for i in cp_tp_part_list], dim=-1)

    return visual_pos_masks_ret, deepstack_visual_embeds_ret
