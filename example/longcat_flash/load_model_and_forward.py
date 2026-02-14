# Test LongCat Flash model: load weights and run forward pass, compare with HF output
#
# Step 1: Generate HF reference output
#   python example/longcat_flash/hf_fwd.py --model_path /path/to/longcat_flash
#
# Step 2: Run Megatron forward pass and compare (single node, smaller variants)
#   torchrun --nproc_per_node=8 example/longcat_flash/load_model_and_forward.py \
#       --model_path /path/to/longcat_flash --tp 8 --pp 1 --ep 1
#
# For 560B model with multiple nodes, use Ray:
#   python example/longcat_flash/launch_forward_with_ray.py \
#       --model_path /path/to/longcat_flash --num_nodes 4 --gpus_per_node 8 \
#       --tp 8 --pp 4 --ep 8

import argparse

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
from mbridge.utils.post_creation_callbacks import freeze_moe_router


HF_OUTPUT_PATH = "/tmp/hf_longcat_flash.pt"


def cos_similarity(a, b):
    """Compute cosine similarity between HF and Megatron outputs."""
    print(f"a {a.shape} b {b.shape}")
    a = a.float()
    a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
    a = a / a.norm(dim=-1, keepdim=True)

    b = b.float()
    b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
    b = b / b.norm(dim=-1, keepdim=True)

    sim = (a * b).sum(dim=-1)
    print(
        f"hf vs megatron cos_similarity min: {sim.min()}; "
        f"max: {sim.max()}; mean: {sim.mean()}"
    )


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


def get_sample_for_forward(hf_model_path, prompt=None):
    """Prepare a sample input for the forward pass."""
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if prompt is None:
        prompt = "A bubble sort in python is "
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    return {"input_ids": input_ids}


def get_args():
    parser = argparse.ArgumentParser(
        description="Load LongCat Flash model and run forward pass"
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
        "--check_export", action="store_true", help="Also check weight export"
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
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for forward pass",
    )
    return parser.parse_args()


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

    # Configure uneven pipeline stages if specified
    if args.num_layers_in_first_pipeline_stage is not None:
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        )
    elif args.pp > 1:
        # Auto-compute uneven PP stage sizes for LongCat Flash
        num_layers = bridge.hf_config.num_hidden_layers
        layers_per_stage = (num_layers + args.pp - 1) // args.pp
        first_last_layers = num_layers - layers_per_stage * (args.pp - 2)
        assert first_last_layers > 1, (
            f"Not enough layers ({num_layers}) for {args.pp} pipeline stages"
        )
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=first_last_layers // 2,
            num_layers_in_last_pipeline_stage=(first_last_layers + 1) // 2,
        )

    model = bridge.get_model(
        post_model_creation_callbacks=[freeze_moe_router], wrap_with_ddp=False
    )
    assert len(model) == 1

    bridge.load_weights(model, hf_model_path, memory_efficient=True)

    # Optionally check export
    if args.check_export:
        print(f"rank{rank}: checking weight export ...")
        keys = bridge.safetensor_io.load_hf_weight_names()
        loaded_keys = set()
        for k, v in bridge.export_weights(model):
            gt = bridge.safetensor_io.load_one_hf_weight(k).cuda()
            assert v.shape == gt.shape, f"mismatch of {k}"
            assert torch.equal(v, gt), f"mismatch of {k}"
            loaded_keys.add(k)
        missing_keys = set(keys) - loaded_keys
        if rank == 0:
            print(f"missing keys: {sorted(missing_keys)}")

    print(f"rank{rank}: end load weight, start forward ...")

    sample = get_sample_for_forward(hf_model_path, args.prompt)
    real_seq_length = sample["input_ids"].shape[-1]
    torch.distributed.barrier()

    with torch.no_grad():
        fwd_bwd_function = get_forward_backward_func()

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

        if mpu.is_pipeline_last_stage():
            megatron_output = mcore_output[0]["logits"]
            if mpu.get_tensor_model_parallel_world_size() > 1:
                megatron_output = gather_from_tensor_model_parallel_region(
                    megatron_output
                )

            megatron_output = megatron_output[:, :real_seq_length, :]

            # Compare with HF output if available
            import os

            if os.path.exists(HF_OUTPUT_PATH):
                hf_output = torch.load(HF_OUTPUT_PATH, map_location="cpu").to(
                    megatron_output.device
                )
                cos_similarity(hf_output, megatron_output)
            else:
                print(
                    f"HF reference output not found at {HF_OUTPUT_PATH}. "
                    f"Run hf_fwd.py first to generate it."
                )
                # Print basic stats instead
                print(
                    f"Megatron output shape: {megatron_output.shape}, "
                    f"mean: {megatron_output.float().mean():.4f}, "
                    f"std: {megatron_output.float().std():.4f}"
                )

            print("Forward pass completed successfully!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()