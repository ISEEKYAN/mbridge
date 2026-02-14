# Test LongCat Flash model: load weights and run token-by-token inference
#
# Usage with torchrun (single node, for smaller variants):
#   torchrun --nproc_per_node=8 example/longcat_flash/load_model_and_inference.py \
#       --model_path /path/to/longcat_flash --tp 8 --pp 1 --ep 1
#
# For the full 560B model across multiple nodes (4 nodes x 8 GPUs = 32 GPUs),
# wrap this logic with Ray (see launch_with_ray.py for the pattern).

import argparse
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tqdm import trange
from transformers import AutoTokenizer

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


def get_args():
    parser = argparse.ArgumentParser(
        description="Load LongCat Flash model and run inference"
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
        "--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A bubble sort in python is ",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the first pipeline stage",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the last pipeline stage",
    )
    return parser.parse_args()


def broadcast_object_within_pp(obj: Any) -> Any:
    """Broadcast an object from the last PP stage to all PP stages."""
    group = mpu.get_pipeline_model_parallel_group()
    if torch.distributed.get_world_size(group) > 1:
        obj_list = [obj]
        torch.distributed.broadcast_object_list(
            obj_list,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=group,
        )
        return obj_list[0]
    else:
        return obj


def mcore_fwd_fn(data_iterator, model):
    """Forward step function for Megatron's pipeline schedule."""
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


def main():
    args = get_args()
    print(f"{args=}")

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
    print(f"rank{rank}: start loading model ...")

    # Load megatron model
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    bridge.config.sequence_parallel = True

    # Configure uneven pipeline stages
    if args.num_layers_in_first_pipeline_stage is not None:
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        )
    elif args.pp > 1:
        num_layers = bridge.hf_config.num_hidden_layers
        layers_per_stage = (num_layers + args.pp - 1) // args.pp
        first_last_layers = num_layers - layers_per_stage * (args.pp - 2)
        assert first_last_layers > 1
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=first_last_layers // 2,
            num_layers_in_last_pipeline_stage=(first_last_layers + 1) // 2,
        )

    model = bridge.get_model(
        post_model_creation_callbacks=[freeze_moe_router], wrap_with_ddp=True
    )
    assert len(model) == 1
    bridge.load_weights(model, hf_model_path)
    print(f"rank{rank}: end load weight, start inference ...")

    # Get EOS token id
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id

    # Prepare input
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").tolist()
    sample = {"input_ids": torch.tensor(input_ids, device=torch.cuda.current_device())}
    generated_tokens = []

    torch.distributed.barrier()
    with torch.no_grad():
        fwd_bwd_function = get_forward_backward_func()

        for i in trange(
            args.max_new_tokens,
            disable=(mpu.get_tensor_model_parallel_rank() != 0),
        ):
            real_seq_length = sample["input_ids"].shape[-1]
            seq_length = real_seq_length
            if real_seq_length % args.tp != 0:
                seq_length = (real_seq_length + args.tp - 1) // args.tp * args.tp
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

            next_token = -1
            if mpu.is_pipeline_last_stage():
                megatron_output = mcore_output[0]["logits"]
                if mpu.get_tensor_model_parallel_world_size() > 1:
                    megatron_output = gather_from_tensor_model_parallel_region(
                        megatron_output
                    )
                megatron_output = megatron_output[:, :real_seq_length, :]
                next_token = megatron_output[:, -1, :].argmax(dim=-1)[0].item()
                if (
                    torch.distributed.get_rank()
                    == torch.distributed.get_world_size() - 1
                ):
                    print(f"{i=} {next_token=}")

            next_token = broadcast_object_within_pp(next_token)
            generated_tokens.append(next_token)
            input_ids[0].append(next_token)
            sample["input_ids"] = torch.tensor(
                input_ids, device=torch.cuda.current_device()
            )
            if next_token == eos_token_id:
                break

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Generated tokens: {generated_tokens}")
        output_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {output_text}")
        print(f"{'='*60}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()