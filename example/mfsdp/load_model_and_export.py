# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Round-trip HF <-> Megatron-FSDP weight conversion verification.

Loads HF weights into a Megatron-FSDP model, then exports back through
``bridge.export_weights`` and compares each tensor with the original HF weight.

Usage:
    torchrun --standalone --nproc_per_node=8 example/mfsdp/load_model_and_export.py \\
        --hf-model /root/models/hf/Qwen2.5-0.5B --tp 2 --ep 1 --cp 1
"""
from __future__ import annotations

import argparse
import sys

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

DDP_CONFIG = {
    "use_distributed_optimizer": True,
    "data_parallel_sharding_strategy": "optim_grads_params",
}


def init_distributed(tp: int, pp: int, cp: int, ep: int, etp: int | None) -> None:
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hf-model", required=True, help="HF model dir or repo id")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--etp", type=int, default=None)
    p.add_argument(
        "--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    p.add_argument(
        "--memory-efficient-load",
        action="store_true",
        help="Stream weights from disk instead of preloading the full mapping.",
    )
    p.add_argument(
        "--hf-out",
        default=None,
        help="If set, also save the exported HF safetensors here (rank-0 only).",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def rank0_print(message: str) -> None:
    if torch.distributed.get_rank() == 0:
        print(message)


def build_mfsdp_model(args: argparse.Namespace):
    dtype = DTYPES[args.torch_dtype]
    bridge = AutoBridge.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code, dtype=dtype
    )
    model = bridge.get_model(
        wrap_with_ddp=True,
        use_megatron_fsdp=True,
        ddp_config=DDP_CONFIG,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        # Megatron-FSDP's broadcast_params() on DTensors hits a missing-DeviceMesh
        # error in some torch versions. The HF load below overwrites every param.
        data_parallel_random_init=False,
    )
    return bridge, model


def compare_exported_weights(bridge, model):
    compared = 0
    skipped_no_gt = 0
    shape_mismatches: list[str] = []
    value_mismatches: list[str] = []
    rank = torch.distributed.get_rank()

    for hf_name, tensor in bridge.export_weights(model):
        if rank != 0:
            continue

        try:
            gt = bridge.safetensor_io.load_one_hf_weight(hf_name)
        except Exception as e:  # noqa: BLE001
            skipped_no_gt += 1
            print(f"[mfsdp-roundtrip] skip {hf_name}: no ground truth ({e})")
            continue

        # Value-model heads may be pruned to one output and have no comparable HF tensor.
        if hf_name == "lm_head.weight" and tensor.shape[0] == 1:
            skipped_no_gt += 1
            continue

        gt = gt.to(tensor.device, dtype=tensor.dtype)
        if tuple(gt.shape) != tuple(tensor.shape):
            shape_mismatches.append(
                f"{hf_name}: exported={tuple(tensor.shape)} hf={tuple(gt.shape)}"
            )
            continue

        if not torch.allclose(tensor, gt, atol=1e-5, rtol=1e-4):
            max_abs = (tensor - gt).abs().max().item()
            value_mismatches.append(f"{hf_name}: max|Δ|={max_abs:.3e}")

        compared += 1

    return compared, skipped_no_gt, shape_mismatches, value_mismatches


def print_summary(
    compared: int,
    skipped_no_gt: int,
    shape_mismatches: list[str],
    value_mismatches: list[str],
) -> None:
    print(
        f"[mfsdp-roundtrip] compared={compared} skipped_no_gt={skipped_no_gt} "
        f"shape_mismatches={len(shape_mismatches)} value_mismatches={len(value_mismatches)}"
    )
    for line in shape_mismatches[:20]:
        print(f"  SHAPE: {line}")
    for line in value_mismatches[:20]:
        print(f"  VALUE: {line}")


def main() -> int:
    args = parse_args()
    init_distributed(args.tp, args.pp, args.cp, args.ep, args.etp)
    rank = torch.distributed.get_rank()
    rank0_print(
        f"[mfsdp-roundtrip] world={torch.distributed.get_world_size()} "
        f"tp={args.tp} pp={args.pp} cp={args.cp} ep={args.ep} etp={args.etp} "
        f"model={args.hf_model}"
    )

    bridge, model = build_mfsdp_model(args)

    rank0_print("[mfsdp-roundtrip] model built; loading HF weights ...")
    bridge.load_weights(
        model,
        args.hf_model,
        memory_efficient=args.memory_efficient_load,
    )
    torch.distributed.barrier()
    rank0_print("[mfsdp-roundtrip] load_weights OK; exporting back to HF ...")

    compared, skipped_no_gt, shape_mismatches, value_mismatches = (
        compare_exported_weights(bridge, model)
    )

    if args.hf_out:
        torch.distributed.barrier()
        rank0_print(f"[mfsdp-roundtrip] saving exported HF model to {args.hf_out} ...")
        bridge.save_weights(model, args.hf_out, memory_efficient=False)

    torch.distributed.barrier()
    failed = False
    if rank == 0:
        print_summary(compared, skipped_no_gt, shape_mismatches, value_mismatches)
        failed = bool(shape_mismatches or value_mismatches)
        if failed:
            print("[mfsdp-roundtrip] FAILED")
        else:
            print("MBRIDGE_MFSDP_ROUNDTRIP_OK")

    status = torch.tensor([int(failed)], device=torch.cuda.current_device())
    torch.distributed.broadcast(status, src=0)
    torch.distributed.destroy_process_group()
    return int(status.item())


if __name__ == "__main__":
    sys.exit(main())
