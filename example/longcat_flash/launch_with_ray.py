#!/usr/bin/env python
"""
Launch LongCat Flash 560B model with Megatron across multiple machines using Ray.

The LongCat Flash model is a 560B MoE model with MLA (Multi-Latent Attention),
requiring at least 32 GPUs (4 nodes x 8 GPUs) with tp=8, pp=4, ep=8.
TP and EP are orthogonal (TP for attention, EP for experts), sharing the same GPUs.

Prerequisites: Ray, NCCL and CUDA are properly installed and configured on every machine.

Start the Ray cluster:
1. On the head node (node-0):
    ray start --head --num-gpus=<GPUs on head> --port=6379
2. On every additional node (node-1 ... node-N):
    ray start --address="node-0-ip:6379" --num-gpus=<GPUs on that node>

Run this script from any machine (typically the head):
    python example/longcat_flash/launch_with_ray.py \
        --model_path /path/to/longcat_flash \
        --num_nodes 4 --gpus_per_node 8 \
        --tp 8 --pp 4 --ep 8

Recommended parallelism configurations for 560B:
  - 32 GPUs (4 nodes):  tp=8, pp=4, ep=8
  - 64 GPUs (8 nodes):  tp=8, pp=4, ep=8 (with dp=2)
"""

import argparse
import os
import socket
from typing import Optional

import ray
import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router


# ---------- Distributed initialization ----------


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp: int,
    pp: int,
    cp: int,
    vpp: Optional[int],
    ep: int,
    etp: Optional[int],
):
    """Initialize torch.distributed according to Megatron's requirements."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = 0
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(backend="nccl")

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


# ---------- Ray worker ----------


@ray.remote(num_gpus=1)
def worker_fn(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    hf_model_path: str,
    tp: int,
    pp: int,
    cp: int,
    vpp: int,
    ep: int,
    etp: Optional[int],
    num_layers_in_first_pipeline_stage: Optional[int] = None,
    num_layers_in_last_pipeline_stage: Optional[int] = None,
    skip_load_weights: bool = False,
):
    """Worker that runs on a single GPU.

    Loads the LongCat Flash model, verifies weight export correctness,
    and optionally skips weight loading for structure-only testing.
    """
    # 1. Initialize distributed environment
    init_distributed(
        rank, world_size, master_addr, master_port, tp, pp, cp, vpp, ep, etp
    )

    # 2. Load model
    bridge = AutoBridge.from_pretrained(hf_model_path)
    if num_layers_in_first_pipeline_stage is not None:
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        )

    model = bridge.get_model(
        post_model_creation_callbacks=[freeze_moe_router], wrap_with_ddp=False
    )

    # Maintain router bias dtype for MoE
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

    print(f"[rank {rank}] Model created successfully.")

    if not skip_load_weights:
        # 3. Load weights
        bridge.load_weights(model, hf_model_path, memory_efficient=True)
        print(f"[rank {rank}] Weights loaded, verifying export ...")

        # 4. Verify weight export
        for k, v in bridge.export_weights(model):
            if torch.distributed.get_rank() != 0:
                continue
            gt = bridge.safetensor_io.load_one_hf_weight(k).to(v.device)
            if k != "lm_head.weight":
                assert v.shape == gt.shape, f"mismatch of {k} {v.shape=} {gt.shape=}"
                assert (
                    v.sum().item() == gt.sum().item()
                ), f"mismatch of {k} {v.sum()=} {gt.sum()=}"
            else:
                if v.shape[0] == 1:
                    print(f"this is a value model, {k} {v.shape=} {gt.shape=}")
            if torch.distributed.get_rank() == 0:
                print(k, "export ok")
    else:
        print(f"[rank {rank}] Skipped weight loading (structure-only test).")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return f"rank {rank} done"


# ---------- Main entry ----------


def main():
    parser = argparse.ArgumentParser(
        description="Launch LongCat Flash 560B model with Ray across multiple nodes"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=4,
        help="Number of physical nodes in the Ray cluster",
    )
    parser.add_argument(
        "--gpus_per_node", type=int, default=8, help="Number of GPUs per node"
    )

    # Megatron parallelism parameters
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=4, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=1, help="Virtual pipeline parallel size"
    )
    parser.add_argument("--ep", type=int, default=8, help="Expert parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
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
    parser.add_argument(
        "--master_port", type=int, default=12355, help="NCCL master port"
    )
    parser.add_argument(
        "--skip_load_weights",
        action="store_true",
        help="Skip loading weights (structure-only test for faster iteration)",
    )
    args = parser.parse_args()

    # Connect to the running Ray cluster
    ray.init()

    world_size = args.num_nodes * args.gpus_per_node
    master_addr = socket.gethostbyname(ray.util.get_node_ip_address())

    print(f"Launching {world_size} workers across {args.num_nodes} nodes ...")
    print(
        f"Parallelism: tp={args.tp}, pp={args.pp}, cp={args.cp}, "
        f"ep={args.ep}, etp={args.etp}, vpp={args.vpp}"
    )

    futures = []
    rank = 0
    for _node_idx in range(args.num_nodes):
        for _local_gpu in range(args.gpus_per_node):
            futures.append(
                worker_fn.remote(
                    rank,
                    world_size,
                    master_addr,
                    args.master_port,
                    args.model_path,
                    args.tp,
                    args.pp,
                    args.cp,
                    args.vpp,
                    args.ep,
                    args.etp,
                    args.num_layers_in_first_pipeline_stage,
                    args.num_layers_in_last_pipeline_stage,
                    args.skip_load_weights,
                )
            )
            rank += 1

    for res in ray.get(futures):
        print(res)

    print("All workers have completed!")


if __name__ == "__main__":
    main()