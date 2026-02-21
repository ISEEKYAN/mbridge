from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig


@dataclass
class Qwen3p5VLTransformerConfig(TransformerConfig):
    patch_size: int = 14
    in_channels: int = 3
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2304
    apply_rotary_pos_emb_in_fp32: bool = False
