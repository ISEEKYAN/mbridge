# Test LongCat Flash 560B model: load weights, export and verify correctness
# This model requires at least 32 GPUs (4 nodes x 8 GPUs) with tp=8, pp=4, ep=8.
# TP and EP are orthogonal (TP for attention, EP for experts), sharing the same GPUs.
#
# Usage with torchrun (single node, for smaller variants or subset parallelism):
#   torchrun --nproc_per_node=8 example/longcat_flash/load_model_and_export.py \
#       --model_path /path/to/longcat_flash --tp 8 --pp 1 --ep 1
#
# For 560B model, use the Ray launcher instead:
#   python example/longcat_flash/launch_with_ray.py \
#       --model_path /path/to/longcat_flash --num_nodes 4 --gpus_per_node 8 \
#       --tp 8 --pp 4 --ep 8

import argparse
import json
import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router


def init_distributed(tp=8, pp=4, cp=1, vpp=1, ep=8, etp=None):
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % 8)
    if pp <= 1:
        vpp = None
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


def compare_parameter_list(parameter_list, hf_model_path):
    """Gather all exported parameter names and compare with HF checkpoint."""
    list_of_parameter_list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(list_of_parameter_list, parameter_list)
    full_parameter_list = set(e for p_list in list_of_parameter_list for e in p_list)

    index_map_file = os.path.join(hf_model_path, "model.safetensors.index.json")
    assert os.path.exists(index_map_file), f"Index file not found: {index_map_file}"
    with open(index_map_file) as f:
        file_mapping = json.load(f)
        hf_parameter_list = set(file_mapping["weight_map"].keys())

    diff1 = full_parameter_list - hf_parameter_list
    diff2 = hf_parameter_list - full_parameter_list

    if torch.distributed.get_rank() == 0:
        if diff1:
            print(f"Extra keys in Megatron but not in HF: {diff1}")
        if diff2:
            print(f"Extra keys in HF but not in Megatron: {diff2}")

    assert not diff1, f"megatron_parameter_list - hf_parameter_list: {diff1}"
    # assert not diff2, f"hf_parameter_list - megatron_parameter_list: {diff2}"


def _maintain_router_bias_dtype(model):
    """Maintain float32 dtype for MoE router expert bias (same as DeepSeekV3)."""
    from mbridge.core.util import unwrap_model

    for m in model:
        m_unwrapped = unwrap_model(m)
        if hasattr(m_unwrapped, "decoder"):
            for layer in m_unwrapped.decoder.layers:
                if (
                    hasattr(layer, "mlp")
                    and hasattr(layer.mlp, "router")
                    and hasattr(layer.mlp.router, "_maintain_float32_expert_bias")
                ):
                    layer.mlp.router._maintain_float32_expert_bias()


def main():
    parser = argparse.ArgumentParser(
        description="Load LongCat Flash model and verify weight export"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor model parallel size")
    parser.add_argument(
        "--pp", type=int, default=4, help="Pipeline model parallel size"
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=None, help="Virtual pipeline model parallel size"
    )
    parser.add_argument("--ep", type=int, default=8, help="Expert model parallel size")
    parser.add_argument(
        "--etp", type=int, default=1, help="Expert tensor parallel size"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save exported weights"
    )
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the first pipeline stage (for uneven PP)",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the last pipeline stage (for uneven PP)",
    )
    args = parser.parse_args()

    # Initialize distributed environment
    init_distributed(
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        vpp=args.vpp,
        ep=args.ep,
        etp=args.etp,
    )

    hf_model_path = args.model_path
    rank = torch.distributed.get_rank()
    print(f"rank{rank}: start loading model")

    # Load model
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    print(f"{bridge.hf_config=}")

    # Set uneven pipeline stage configuration if specified
    if args.num_layers_in_first_pipeline_stage is not None:
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        )
    model = bridge.get_model(
        post_model_creation_callbacks=[freeze_moe_router], wrap_with_ddp=False
    )

    # Maintain router bias dtype for MoE
    _maintain_router_bias_dtype(model)

    # Print model parameter names on rank 0
    if rank == 0:
        for k, v in model[0].named_parameters():
            print(f"{k} => {v.shape}")

    print(f"rank{rank}: start loading weights from {hf_model_path}")
    bridge.load_weights(model, hf_model_path)
    print(f"rank{rank}: end load weight")

    # Export weights and compare values
    parameter_list = []
    not_matched_keys = set()
    for k, v in bridge.export_weights(model):
        if rank != 0:
            parameter_list.append(k)
            continue
        gt = bridge.safetensor_io.load_one_hf_weight(k).to(v.device)
        if k != "lm_head.weight":
            assert v.shape == gt.shape, f"mismatch of {k} {v.shape=} {gt.shape=}"
            v_sum = v.sum()
            gt_sum = gt.sum()
            if v_sum.item() != gt_sum.item():
                not_matched_keys.add(k)
                print(
                    f"mismatch of {k}, {v_sum} vs {gt_sum}, "
                    f"{v.device} vs {gt.device}, {v.dtype} vs {gt.dtype}"
                )
        else:
            if v.shape[0] == 1:
                print(f"this is a value model, {k} {v.shape=} {gt.shape=}")
        print(k, "export ok")
        parameter_list.append(k)

    # Compare parameter list with HF checkpoint
    compare_parameter_list(parameter_list, hf_model_path)

    if rank == 0:
        print(f"not_matched_keys: {not_matched_keys}")

    if args.save_path:
        bridge.save_weights(model, args.save_path, memory_efficient=False)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()