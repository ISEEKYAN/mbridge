import dataclasses
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


def load_some_hf_weight(hf_dir: str, hf_weight_names: list[str]) -> dict:
    """
    加载huggingface的权重
    """
    import json
    import os
    from glob import glob

    from safetensors import safe_open

    # 检查是否存在index文件
    index_file = os.path.join(hf_dir, "model.safetensors.index.json")

    @lru_cache(maxsize=None)
    def load_index_file(index_file: str) -> dict:
        if not os.path.exists(index_file):
            return {}
        with open(index_file, "r") as f:
            index = json.load(f)["weight_map"]
        return index

    ret = {}
    index = load_index_file(index_file)
    if index:
        file_to_weight_map = defaultdict(list)
        for name in hf_weight_names:
            filename = index[name]
            file_to_weight_map[filename].append(name)
        for filename, weight_names in file_to_weight_map.items():
            safetensor_file = os.path.join(hf_dir, filename)
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for name in weight_names:
                    ret[name] = f.get_tensor(name)
        return ret
    print("warning: 未找到index文件，将搜索所有safetensors文件")

    # 搜索所有safetensors文件
    safetensor_files = glob(os.path.join(hf_dir, "*.safetensors"))
    print(safetensor_files)
    # 如果有safetensors文件
    if safetensor_files:
        # 遍历每个safetensors文件
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                to_load = set(hf_weight_names) & set(f.keys())
                if to_load:
                    for name in to_load:
                        ret[name] = f.get_tensor(name)
        if len(ret) != len(hf_weight_names):
            raise ValueError(
                f"在{hf_dir}的safetensors文件中未找到权重{set(hf_weight_names)-set(ret.keys())}"
            )
        return ret
    raise ValueError(f"在{hf_dir}的safetensors文件中未找到权重{hf_weight_names}")


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
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                this_model = model_provider_func(
                    pre_process=pre_process, post_process=post_process
                )
                this_model.model_type = model_type
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
        if optimizer_config:
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
