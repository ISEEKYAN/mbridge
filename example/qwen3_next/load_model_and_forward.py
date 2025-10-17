# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 load_model_and_export.py --model_path /path/to/model


import argparse
import json
import os
from typing import List
import requests

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from megatron.core import parallel_state
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router


# hf logits vs megatron logits
def cos_similarity(a, b):
    print(f"a {a.shape} b {b.shape}")
    a = a.to(b.device)
    a = a.float()
    # a = a / a.norm(dim=-1, keepdim=True)
    a = torch.exp(a)
    a = a / a.norm(dim=-1, keepdim=True)
    """
    a = (a - a.mean(dim=-1, keepdim=True)) 
    a = a / a.norm(dim=-1, keepdim=True)
    """
    b = b.float()
    # b =  b / b.norm(dim=-1, keepdim=True)
    b = torch.exp(b)
    b = b / b.norm(dim=-1, keepdim=True)
    """
    b = (b - b.mean(dim=-1, keepdim=True)) 
    b =  b / b.norm(dim=-1, keepdim=True)
    """
    sim = (a * b).sum(dim=-1)
    print(
        f"hf vs megatron cos_similarity min: {sim.min()}; max: {sim.max()}; mean: {sim.mean()}"
    )


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    pad_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    pad_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0
    if pad_mask_loss:
        loss_mask[data == pad_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token] & position_ids[b, data[b] == pad_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def is_first_rank():
    """First tensor and pipeline parallel rank."""
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def init_distributed(tp=2, pp=1, cp=1, vpp=1, ep=1, etp=None):
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
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


def get_args():
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor model parallel size")
    parser.add_argument(
        "--pp", type=int, default=1, help="Pipeline model parallel size"
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=None, help="Virtual pipeline model parallel size"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert model parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save weights"
    )
    args = parser.parse_args()
    return args


def main(args):
    # Parse command line arguments
    # Initialize distributed environment
    init_distributed(
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        vpp=args.vpp,
        ep=args.ep,
        etp=args.etp,
    )

    # Load megatron model
    hf_model_path = args.model_path
    print(f"rank{torch.distributed.get_rank()}: {args=} start loading model ...")
    bridge = AutoBridge.from_pretrained(hf_model_path)
    bridge.config.sequence_parallel = True if args.tp > 1 else False
    model = bridge.get_model()
    # if torch.distributed.get_rank() == 0:
    #     print(f"Model arch {model} len {len(model)}")

    # torch.distributed.barrier()
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")
    torch.distributed.barrier()
    for pname, params in model[0].named_parameters():
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            print(f"Trace export_weights {pname=} shape {params.shape=} dtype {params.dtype=} {params.sum()}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    prompt = "李白，字太白，号"
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer([text], return_tensors="pt")["input_ids"]

    attn_mask, _, pids = get_ltor_masks_and_position_ids(
        input_ids, None, tokenizer.pad_token_id, False, False, False, True
    )
    sample_list = [{"input_ids": input_ids, "attention_mask": attn_mask, "position_ids": pids}]
    print(f"model input {input_ids.shape=} {input_ids=} {attn_mask.shape=} {pids.shape=}"
          f" {attn_mask=} {pids=}")

    with torch.no_grad():
        fwd_bwd_function = get_forward_backward_func()
        real_seq_length = input_ids.shape[-1]
        seq_length = real_seq_length
        if real_seq_length % args.tp != 0:
            seq_length = (real_seq_length + args.tp - 1) // args.tp * args.tp
            sample_list[0]["input_ids"] = F.pad(
                sample_list[0]["input_ids"],
                (0, seq_length - real_seq_length, 0, 0),
                value=0,
            )

        def mcore_fwd_fn(data_iter, model):
            sample = next(data_iter)

            output_tensor = model(
                input_ids=sample['input_ids'].cuda(),
                position_ids=sample['position_ids'].cuda(),
                attention_mask=sample['attention_mask'].cuda(),
            )
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            assert isinstance(output_tensor, torch.Tensor)
            def loss_func(output_tensor, non_loss_data=True):
                loss = output_tensor.mean()
                return loss, {
                    "loss": loss.detach(),
                    "logits": output_tensor.detach(),
                }
            return output_tensor, loss_func

        mcore_output = fwd_bwd_function(
            forward_step_func=mcore_fwd_fn,
            data_iterator=iter(sample_list),
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_length,
            decoder_seq_length=seq_length,
            micro_batch_size=1,
        )

        if mpu.is_pipeline_last_stage():
            megatron_output = mcore_output[0]["logits"]
            if mpu.get_tensor_model_parallel_world_size() > 1:
                megatron_output = gather_from_tensor_model_parallel_region(
                    megatron_output
                )
            megatron_output = megatron_output[:, :real_seq_length, :]
            torch.save(megatron_output, f"./megatron_qwen3next_tp{args.tp}.pt")

            # hf_output = torch.load("./hf_qwen3next.pt")
            # cos_similarity(hf_output, megatron_output)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    args = get_args()
    main(args)
