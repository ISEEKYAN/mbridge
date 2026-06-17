# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions used throughout Megatron core"""

import warnings
import torch
from megatron.core import parallel_state

def get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
    """Issue a deprecation warning if tp_group is None and return the default tp group."""
    # TODO(zijiey): remove this function later.
    if not torch.distributed.is_initialized():
        return None

    if tp_group is None:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            warnings.warn(
                "Warning: tp_group is None, using default tp group. "
                "Passing tp_group will be mandatory soon",
                DeprecationWarning,
                stacklevel=2,
            )
        if is_expert:
            tp_group = parallel_state.get_expert_tensor_parallel_group(
                check_initialized=check_initialized
            )
        else:
            tp_group = parallel_state.get_tensor_model_parallel_group(
                check_initialized=check_initialized
            )
    return tp_group
