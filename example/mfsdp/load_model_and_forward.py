# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Load HF weights into a Megatron-FSDP-wrapped model and decode a few tokens. It supports TP/CP/EP with PP fixed to 1.

Usage:
    torchrun --standalone --nproc_per_node=8 \\
        example/mfsdp/load_model_and_forward.py \\
        --hf-model /root/models/hf/Qwen2.5-0.5B --tp 2 --cp 2 --max-new-tokens 16
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer

from mbridge import AutoBridge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hf-model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--etp", type=int, default=None)
    p.add_argument("--prompt", default="A bubble sort in python is ")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    p.add_argument("--memory-efficient-load", action="store_true")
    return p.parse_args()


def init_distributed(tp: int, pp: int, cp: int, ep: int, etp: Optional[int]) -> None:
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


def gather_output_from_context_parallel(
    input_: torch.Tensor, seq_dim: int
) -> torch.Tensor:
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return input_

    assert seq_dim in (0, 1) and input_.dim() > seq_dim
    input_ = input_.view(
        *input_.shape[:seq_dim],
        2,
        input_.shape[seq_dim] // 2,
        *input_.shape[(seq_dim + 1) :],
    )

    gathered = [torch.zeros_like(input_) for _ in range(cp_size)]
    torch.distributed.all_gather(
        gathered, input_, group=mpu.get_context_parallel_group()
    )

    reordered = [None for _ in range(2 * cp_size)]
    if seq_dim == 1:
        for rank in range(cp_size):
            reordered[rank] = gathered[rank][:, 0]
            reordered[2 * cp_size - rank - 1] = gathered[rank][:, 1]
    else:
        for rank in range(cp_size):
            reordered[rank] = gathered[rank][0]
            reordered[2 * cp_size - rank - 1] = gathered[rank][1]
    return torch.cat(reordered, dim=seq_dim)


def next_multiple(value: int, factor: int) -> int:
    return ((value + factor - 1) // factor) * factor


def forward_step(data_iterator, model):
    sample = next(data_iterator)
    position_ids = sample.get("position_ids")
    if position_ids is not None:
        position_ids = position_ids.cuda()
    output = model(
        input_ids=sample["input_ids"].cuda(),
        position_ids=position_ids,
        attention_mask=None,
        runtime_gather_output=False,
    )
    if isinstance(output, tuple):
        output = output[0]

    def loss_fn(output_tensor, non_loss_data=True):
        loss = output_tensor.mean()
        return loss, {"logits": output_tensor.detach()}

    return output, loss_fn


def main() -> int:
    args = parse_args()
    if args.pp != 1:
        raise ValueError("This generation smoke test supports PP=1 only.")
    init_distributed(args.tp, args.pp, args.cp, args.ep, args.etp)
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    if rank == 0:
        print(
            f"[mfsdp-generate] world={world} tp={args.tp} pp={args.pp} "
            f"cp={args.cp} ep={args.ep} etp={args.etp} model={args.hf_model}"
        )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    bridge = AutoBridge.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code, dtype=dtype
    )
    model = bridge.get_model(
        wrap_with_ddp=True,
        use_megatron_fsdp=True,
        ddp_config={
            "use_distributed_optimizer": True,
            "data_parallel_sharding_strategy": "optim_grads_params",
        },
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        data_parallel_random_init=False,
    )
    if rank == 0:
        print("[mfsdp-generate] model built; loading HF weights ...")
    bridge.load_weights(
        model, args.hf_model, memory_efficient=args.memory_efficient_load
    )
    torch.distributed.barrier()
    if rank == 0:
        print("[mfsdp-generate] load_weights OK; generating ...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code
    )
    uses_multimodal_rope = (
        getattr(bridge.hf_config, "vision_config", None) is not None
        and getattr(bridge.hf_config, "image_token_id", None) is not None
    )

    generated: list[int] = []
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").tolist()
    seq_length_factor = args.tp
    if args.cp > 1:
        seq_length_factor *= args.cp * 2

    for model_chunk in model:
        model_chunk.eval()

    fwd_bwd_function = get_forward_backward_func()
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            real_seq_length = len(input_ids[0])
            seq_length = next_multiple(real_seq_length, seq_length_factor)
            cur_input_ids = torch.tensor(input_ids, device=torch.cuda.current_device())
            if seq_length > real_seq_length:
                cur_input_ids = F.pad(
                    cur_input_ids, (0, seq_length - real_seq_length), value=0
                )
            cur_position_ids = torch.arange(
                seq_length, device=cur_input_ids.device, dtype=torch.long
            ).unsqueeze(0)
            sample = {"input_ids": cur_input_ids}
            if not uses_multimodal_rope:
                sample["position_ids"] = cur_position_ids

            output = fwd_bwd_function(
                forward_step_func=forward_step,
                data_iterator=iter([sample]),
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=seq_length,
                decoder_seq_length=seq_length,
                micro_batch_size=1,
            )

            logits = output[0]["logits"]
            logits = gather_output_from_context_parallel(logits, seq_dim=1)
            if mpu.get_tensor_model_parallel_world_size() > 1:
                logits = gather_from_tensor_model_parallel_region(logits)
            logits = logits[:, :real_seq_length, :]

            next_tok = logits[:, -1, :].argmax(dim=-1)
            if rank == 0:
                print(f"[mfsdp-generate] step={step} next_token={int(next_tok.item())}")
            tok_id = int(next_tok.item())
            generated.append(tok_id)
            if tok_id == (tokenizer.eos_token_id or -1):
                break
            input_ids[0].append(tok_id)

    if rank == 0:
        print(f"PROMPT: {args.prompt}")
        print(f"COMPLETION: {tokenizer.decode(generated)}")
        print("MBRIDGE_MFSDP_GENERATE_OK")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
