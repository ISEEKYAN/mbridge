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



def reorganize_inputs(
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor = None,
    pixel_values_videos: torch.Tensor = None,
    image_grid_thw: torch.Tensor = None,
    video_grid_thw: torch.Tensor = None,
    image_input_mask: torch.Tensor = None,
    video_input_mask: torch.Tensor = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    square_merge_size: int = 4,
):
    if pixel_values is None:
        if video_input_mask is None and pixel_values_videos is not None:
            video_input_mask = (input_ids == video_token_id).contiguous()
        return pixel_values_videos, video_grid_thw, video_input_mask

    if pixel_values_videos is None:
        if image_input_mask is None and pixel_values is not None:
            image_input_mask = (input_ids == image_token_id).contiguous()
        return pixel_values, image_grid_thw, image_input_mask

    image_thw_cpu = image_grid_thw.tolist()
    video_thw_cpu = video_grid_thw.tolist()
    vision_indexs = find_vision_id_index(
        input_ids.view(-1), image_token_id, video_token_id
    )
    len_split = sum([thw[0] for thw in image_thw_cpu])
    len_split += sum([thw[0] for thw in video_thw_cpu])
    assert len_split == len(vision_indexs)

    vision_values = []
    vision_grid_thw = []
    idx = 0
    video_idx = 0
    image_idx = 0
    video_seqlen = 0
    image_seqlen = 0
    while idx < len(vision_indexs):
        start, end, token_id = vision_indexs[idx]
        if token_id == image_token_id:
            seqlen = 0
            thw = image_thw_cpu[image_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == image_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(pixel_values[image_seqlen : (image_seqlen + seqlen)])
            vision_grid_thw.append(thw)

            image_idx += 1
            idx += thw[0]
            image_seqlen += seqlen
        elif token_id == video_token_id:
            seqlen = 0
            thw = video_thw_cpu[video_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == video_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(
                pixel_values_videos[video_seqlen : (video_seqlen + seqlen)]
            )
            vision_grid_thw.append(thw)

            video_idx += 1
            idx += thw[0]
            video_seqlen += seqlen
        else:
            assert False, f"should not have {token_id=}"

    if video_input_mask is None:
        video_input_mask = input_ids == video_token_id

    if image_input_mask is None:
        image_input_mask = input_ids == image_token_id

    vision_values = torch.cat(vision_values)
    vision_grid_thw = torch.tensor(
        vision_grid_thw, device=image_grid_thw.device, dtype=image_grid_thw.dtype
    )
    vision_input_mask = video_input_mask | image_input_mask

    return vision_values, vision_grid_thw, vision_input_mask
