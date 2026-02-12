#!/usr/bin/env python
"""
Launch LongCat Flash 560B forward pass test across multiple machines using Ray.

This combines the Ray multi-node launch pattern with the forward pass test.

Prerequisites: Ray cluster running across all nodes.

Usage:
    # Step 1: Generate HF reference output (on a machine with enough memory)
    python example/longcat_flash/hf_fwd.py --model_path /path/to/longcat_flash

    # Step 2: Launch multi-node forward pass (4 nodes x 8 GPUs = 32 GPUs)
    python example/longcat_flash/launch_forward_with_ray.py \
        --model_path /path/to/longcat_flash \
        --num_nodes 4 --gpus_per_node 8 \
        --tp 8 --pp 4 --ep 8
"""

import argparse
import os
import socket
from typing import Optional

import ray
import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router

import torch.nn.functional as F


HF_OUTPUT_PATH = "/tmp/hf_longcat_flash.pt"


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


def mcore_fwd_fn(data_iterator, model):
    sample = next(data_iterator)
    output_tensor = model(
        input_ids=sample["input_ids"].cuda(),
        position_ids=None,
        attention_mask=None,
    )
    if isinstance(output_tensor, tuple):
        output_tensor = output_tensor[0]
    assert isinstance(output_tensor, torch.Tensor)

    def loss_fn(output_tensor, non_loss_data=True):
        loss = output_tensor.mean()
        return loss, {
            "loss": loss.detach(),
            "logits": output_tensor.detach(),
        }

    return output_tensor, loss_fn


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
    prompt: str = "A bubble sort in python is ",
):
    # 1. Initialize
    init_distributed(
        rank, world_size, master_addr, master_port, tp, pp, cp, vpp, ep, etp
    )

    # 2. Load model
    bridge = AutoBridge.from_pretrained(hf_model_path)
    bridge.config.sequence_parallel = True

    if num_layers_in_first_pipeline_stage is not None:
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        )

    model = bridge.get_model(
        post_model_creation_callbacks=[freeze_moe_router], wrap_with_ddp=False
    )
    assert len(model) == 1
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"[rank {rank}] Weights loaded, running forward pass ...")

    # 3. Prepare input
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    sample = {"input_ids": input_ids}
    real_seq_length = input_ids.shape[-1]

    # 4. Forward pass
    torch.distributed.barrier()
    with torch.no_grad():
        fwd_bwd_function = get_forward_backward_func()

        seq_length = real_seq_length
        if real_seq_length % tp != 0:
            seq_length = (real_seq_length + tp - 1) // tp * tp
            sample["input_ids"] = F.pad(
                sample["input_ids"],
                (0, seq_length - real_seq_length, 0, 0),
                value=0,
            )

        mcore_output = fwd_bwd_function(
            forward_step_func=mcore_fwd_fn,
            data_iterator=iter([sample]),
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_length,
            decoder_seq_length=seq_length,
            micro_batch_size=1,
        )

        result_msg = f"rank {rank} forward done"
        if mpu.is_pipeline_last_stage():
            megatron_output = mcore_output[0]["logits"]
            if mpu.get_tensor_model_parallel_world_size() > 1:
                megatron_output = gather_from_tensor_model_parallel_region(
                    megatron_output
                )
            megatron_output = megatron_output[:, :real_seq_length, :]

            if os.path.exists(HF_OUTPUT_PATH):
                hf_output = torch.load(HF_OUTPUT_PATH, map_location="cpu").to(
                    megatron_output.device
                )
                a = hf_output.float()
                a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
                a = a / a.norm(dim=-1, keepdim=True)
                b = megatron_output.float()
                b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
                b = b / b.norm(dim=-1, keepdim=True)
                sim = (a * b).sum(dim=-1)
                result_msg = (
                    f"rank {rank} forward done | "
                    f"cos_sim min={sim.min():.4f} max={sim.max():.4f} mean={sim.mean():.4f}"
                )
            else:
                result_msg = (
                    f"rank {rank} forward done | "
                    f"output shape={megatron_output.shape} "
                    f"mean={megatron_output.float().mean():.4f}"
                )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return result_msg


def main():
    parser = argparse.ArgumentParser(
        description="Launch LongCat Flash forward pass with Ray"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument("--num_nodes", type=int, default=4, help="Number of nodes")
    parser.add_argument(
        "--gpus_per_node", type=int, default=8, help="GPUs per node"
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=4, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--vpp", type=int, default=1, help="Virtual pipeline parallel size")
    parser.add_argument("--ep", type=int, default=8, help="Expert parallel size")
    parser.add_argument("--etp", type=int, default=None, help="Expert tensor parallel size")
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage", type=int, default=None,
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage", type=int, default=None,
    )
    parser.add_argument("--master_port", type=int, default=12355, help="NCCL master port")
    parser.add_argument("--prompt", type=str, default="A bubble sort in python is ")
    args = parser.parse_args()

    ray.init()
    world_size = args.num_nodes * args.gpus_per_node
    master_addr = socket.gethostbyname(ray.util.get_node_ip_address())

    print(f"Launching {world_size} workers for forward pass ...")

    futures = []
    rank = 0
    for _node_idx in range(args.num_nodes):
        for _local_gpu in range(args.gpus_per_node):
            futures.append(
                worker_fn.remote(
                    rank, world_size, master_addr, args.master_port,
                    args.model_path, args.tp, args.pp, args.cp, args.vpp,
                    args.ep, args.etp,
                    args.num_layers_in_first_pipeline_stage,
                    args.num_layers_in_last_pipeline_stage,
                    args.prompt,
                )
            )
            rank += 1

    for res in ray.get(futures):
        print(res)

    print("Forward pass test completed!")


if __name__ == "__main__":
    main()