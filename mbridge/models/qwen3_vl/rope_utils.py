# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.
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


from __future__ import annotations

import logging
from typing import List, Optional

import torch
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    get_pos_emb_on_this_cp_rank,
)
from torch import Tensor, nn

from mbridge.models.qwen3_vl.transformer_config import Qwen3VLTransformerConfig

# Prefer fused RoPE from Apex as we need the `transpose_output_memory` argument for the bshd trick.
# See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2469.
try:
    # pylint: disable=unused-import
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
except ImportError:
    fused_apply_rotary_pos_emb = None


logger = logging.getLogger(__name__)


class Qwen3VLMultimodalRotaryEmbedding(nn.Module):
    """Multimodal Rotary Embedding for language model.
    only support for qwen3vl

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: int = 10000,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved
        assert not self.rotary_interleaved, "only support qwen3vl"

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base
            ** (
                torch.arange(
                    0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()
                )
                / dim
            )
        )

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids: torch.Tensor, mrope_section: List[int]) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(
            3, seq.shape[1], -1, 1
        )
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(
                emb, 0, parallel_state.get_context_parallel_group()
            )
        return emb


# Slightly modified from Qwen3VLModel.get_rope_index
def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def apply_rotary_pos_emb_thd_absolute(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    return _apply_rotary_pos_emb_bshd(
        t[:, None], freqs, rotary_interleaved=rotary_interleaved
    ).squeeze(1)


def apply_rotary_pos_emb_absolute(
    t: Tensor,
    freqs: Tensor,
    config: Qwen3VLTransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    bshd (conventional) / thd (packed seq) format

    In Qwen3-VL, the shape of freqs is (seq_length, bs, 1, 2 * dim) instead of [max_seqlen, 1, 1, 2 * dim]
    """
    assert not config.apply_rope_fusion
    orig_t_dtype = t.dtype
    if config.apply_rotary_pos_emb_in_fp32:
        t = t.float()

    if cu_seqlens is None:
        result = _apply_rotary_pos_emb_bshd(
            t, freqs, rotary_interleaved=config.rotary_interleaved
        )
    else:
        result = apply_rotary_pos_emb_thd_absolute(
            t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved
        )

    if config.apply_rotary_pos_emb_in_fp32:
        result = result.to(orig_t_dtype)

    return result
