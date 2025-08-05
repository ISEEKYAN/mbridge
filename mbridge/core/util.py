# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import inspect
import json
import os
from collections import defaultdict
from functools import lru_cache

import torch
from megatron.core import mpu
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import (
    StragglerDetector,
    check_param_hashes_across_dp_replicas,
    get_model_config,
    is_te_min_version,
)


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    fp16: bool = False,
    bf16: bool = True,
    virtual_pipeline_model_parallel_size: int = None,
    encoder_pipeline_model_parallel_size: int = 0,
    use_torch_fsdp2: bool = False,
    use_custom_fsdp: bool = False,
    use_precision_aware_optimizer: bool = False,
    use_cpu_initialization: bool = False,
    init_model_with_meta_device: bool = False,
    overlap_param_gather_with_optimizer_step: bool = False,
    data_parallel_random_init: bool = True,
    ddp_config: dict = None,
    optimizer_config: dict = None,
):
    """Build the model.
    copied from megatron/training/training.py but remove args
    """

    # Build model.
    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and virtual_pipeline_model_parallel_size is not None
        ):
            if model_type == ModelType.encoder_and_decoder:
                assert (
                    encoder_pipeline_model_parallel_size == 0
                ), "Interleaved schedule not supported for model with encoder on separate PP rank"
            model = []
            for i in range(virtual_pipeline_model_parallel_size):
                # Set pre_process and post_process only after virtual rank is set.
                if (
                    "vp_stage"
                    in inspect.signature(mpu.is_pipeline_first_stage).parameters
                ):
                    pre_process = mpu.is_pipeline_first_stage(
                        ignore_virtual=False, vp_stage=i
                    )
                    post_process = mpu.is_pipeline_last_stage(
                        ignore_virtual=False, vp_stage=i
                    )
                else:
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                this_model = model_provider_func(
                    pre_process=pre_process, post_process=post_process, vp_stage=i
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            add_encoder = True
            add_decoder = True
            if model_type == ModelType.encoder_and_decoder:
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    rank = mpu.get_pipeline_model_parallel_rank()
                    first_decoder_rank = encoder_pipeline_model_parallel_size
                    world_size = mpu.get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == first_decoder_rank
                    post_process = (rank == (first_decoder_rank - 1)) or (
                        rank == (world_size - 1)
                    )
                    add_encoder = mpu.is_inside_encoder(rank)
                    add_decoder = mpu.is_inside_decoder(rank)
                model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder,
                )
            else:
                model = model_provider_func(
                    pre_process=pre_process, post_process=post_process
                )
            model.model_type = model_type
        return model

    if init_model_with_meta_device:
        with torch.device("meta"):
            model = build_model()
    else:
        model = build_model()
    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )

    # Print number of parameters.
    num_parameters = sum(
        [
            sum([p.nelement() for p in model_module.parameters()])
            for model_module in model
        ]
    )
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                num_parameters,
            ),
            flush=True,
        )

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not (use_torch_fsdp2 and use_cpu_initialization)
        and not init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if fp16 or bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        from megatron.core.distributed import DistributedDataParallelConfig

        if use_torch_fsdp2:
            try:
                from megatron.core.distributed import (
                    TorchFullyShardedDataParallel as torch_FSDP,
                )

                HAVE_FSDP2 = True
            except ImportError:
                HAVE_FSDP2 = False
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        elif use_custom_fsdp:
            from megatron.core.distributed.custom_fsdp import (
                FullyShardedDataParallel as custom_FSDP,
            )

            DP = custom_FSDP
        else:
            from megatron.core.distributed import DistributedDataParallel as DDP

            DP = DDP

        config = get_model_config(model[0])

        # default
        kwargs = {"grad_reduce_in_fp32": True, "use_distributed_optimizer": True}
        if ddp_config is not None:
            kwargs.update(ddp_config)
        if optimizer_config is not None:
            import warnings

            warnings.warn(
                "optimizer_config is deprecated to set DistributedDataParallelConfig, use ddp_config instead",
                DeprecationWarning,
            )
            kwargs.update(optimizer_config)
        if use_custom_fsdp and use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False

        ddp_config = DistributedDataParallelConfig(**kwargs)

        if not use_torch_fsdp2:
            # In the custom FSDP and DDP use path, we need to initialize the bucket size.

            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000,
                    1000000
                    * mpu.get_data_parallel_world_size(with_context_parallel=True),
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [
            DP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0)
                or overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


from megatron.core import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import (
    FullyShardedDataParallel as custom_FSDP,
)

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, custom_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, custom_FSDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def broadcast_from_megatron_pp(tensor: torch.Tensor):
    # tensor is not None only in one of the pp ranks
    if tensor is not None:
        shape = tensor.shape
        dtype = tensor.dtype
        tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
        partition_dim = getattr(tensor, "partition_dim", None)
        tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
    else:
        tensor_spec = None
    tensor_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=tensor_spec_output,
        obj=tensor_spec,
        group=mpu.get_pipeline_model_parallel_group(),
    )
    # find the src rank
    target_tensor_spec = None
    src_rank = None
    for rank, tensor_spec in enumerate(tensor_spec_output):
        if tensor_spec is not None:
            if target_tensor_spec is None:
                target_tensor_spec = tensor_spec
            else:
                raise ValueError("A tensor exists on two pp ranks")
            src_rank = rank
    assert target_tensor_spec is not None
    if tensor is None:
        tensor = torch.empty(
            size=target_tensor_spec[0],
            dtype=target_tensor_spec[1],
            device=torch.cuda.current_device(),
        )
        if target_tensor_spec[2] is not None:
            tensor.tensor_model_parallel = target_tensor_spec[2]
        if target_tensor_spec[3] is not None:
            tensor.partition_dim = target_tensor_spec[3]

    global_rank = torch.distributed.get_global_rank(
        group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank
    )
    torch.distributed.broadcast(
        tensor=tensor, src=global_rank, group=mpu.get_pipeline_model_parallel_group()
    )
    return tensor


def broadcast_str_from_megatron_pp(obj: any) -> any:
    obj_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=obj_output, obj=obj, group=mpu.get_pipeline_model_parallel_group()
    )

    src_rank = None
    target_obj = None
    for rank, item in enumerate(obj_output):
        if item is not None:
            if target_obj is not None:
                raise ValueError("An object exists on two pp ranks")
            target_obj = item
            src_rank = rank

    assert target_obj is not None, "No valid object found to broadcast."

    global_rank = torch.distributed.get_global_rank(
        group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank
    )

    obj_output = [None] * torch.distributed.get_world_size(
        group=mpu.get_pipeline_model_parallel_group()
    )
    obj_output[0] = target_obj
    torch.distributed.broadcast_object_list(
        object_list=obj_output,
        src=global_rank,
        group=mpu.get_pipeline_model_parallel_group(),
    )

    return obj_output[0]
