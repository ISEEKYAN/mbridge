"""Create a Qwen3.5 model with mbridge and save it back to HF format.

Usage:
  torchrun --nproc_per_node=8 example/qwen3_5/test_save_mtp.py \
      --model_path hf-hub/Qwen/Qwen3.5-4B \
      --save_path qwen3_5_4b_saved \
      --tp 2 --pp 2 --cp 1 --ep 1
"""

import argparse

import torch
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge
from mbridge.core.safetensor_io import SafeTensorIO


def init_distributed(
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    vpp: int | None = None,
    ep: int = 1,
    etp: int | None = None,
) -> None:
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Qwen3.5 model with mbridge and save it to HF format."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=None, help="Virtual pipeline parallel size"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory-efficient HF safetensor saving.",
    )
    return parser.parse_args()


def set_pipeline_stage_args(bridge, pp: int) -> None:
    if pp <= 1:
        return

    num_layers = bridge.hf_config.text_config.num_hidden_layers
    first_last_layer = num_layers - (num_layers + pp - 1) // pp * (pp - 2)
    assert first_last_layer > 1
    bridge.set_extra_args(
        num_layers_in_first_pipeline_stage=first_last_layer // 2,
        num_layers_in_last_pipeline_stage=(first_last_layer + 1) // 2,
    )


def compare_saved_weights(model_path: str, save_path: str) -> None:
    source_io = SafeTensorIO(model_path)
    saved_io = SafeTensorIO(save_path)

    source_keys = set(source_io.load_hf_weight_names())
    saved_keys = set(saved_io.load_hf_weight_names())
    assert source_keys == saved_keys, (
        f"saved HF keys mismatch: "
        f"missing={sorted(source_keys - saved_keys)}, "
        f"unexpected={sorted(saved_keys - source_keys)}"
    )

    mtp_keys = sorted(key for key in source_keys if "mtp" in key)
    assert mtp_keys, "no MTP weights found in the source HF model"

    for key in sorted(source_keys):
        source_weight = source_io.load_one_hf_weight(key)
        saved_weight = saved_io.load_one_hf_weight(key)
        assert source_weight.shape == saved_weight.shape, (
            f"shape mismatch for {key}: "
            f"source={source_weight.shape}, saved={saved_weight.shape}"
        )
        source_weight = source_weight.to(saved_weight.dtype)
        if torch.equal(source_weight, saved_weight):
            continue
        if source_weight.is_floating_point() and torch.allclose(
            source_weight, saved_weight, atol=1e-3, rtol=1e-3
        ):
            continue
        assert False, f"value mismatch for {key}"

    print(
        f"All {len(source_keys)} HF weights match exactly or within tolerance "
        f"after casting source to saved dtype, "
        f"including {len(mtp_keys)} MTP weights."
    )


def main() -> None:
    args = get_args()
    init_distributed(
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        vpp=args.vpp,
        ep=args.ep,
        etp=args.etp,
    )

    rank = torch.distributed.get_rank()
    if rank == 0:
        print(f"Loading Qwen3.5 model from {args.model_path}")

    bridge = AutoBridge.from_pretrained(args.model_path)
    bridge.config.sequence_parallel = True
    set_pipeline_stage_args(bridge, args.pp)

    model = bridge.get_model(model_type=ModelType.encoder_or_decoder)
    assert len(model) == 1
    bridge.load_weights(model, args.model_path, memory_efficient=False)

    if rank == 0:
        print(f"Saving Qwen3.5 model to {args.save_path}")
    bridge.save_weights(
        model,
        args.save_path,
        memory_efficient=args.memory_efficient,
    )

    torch.distributed.barrier()
    if rank == 0:
        print("Save finished.")
        print("Comparing saved HF weights with the source HF weights.")
        compare_saved_weights(args.model_path, args.save_path)

    distributed_save_path = f"{args.save_path}_distributed_filesystem"
    if rank == 0:
        print(f"Saving Qwen3.5 model to {distributed_save_path} with DFS fast path")
    bridge.save_weights(
        model,
        distributed_save_path,
        memory_efficient=True,
        distributed_filesystem=True,
    )

    torch.distributed.barrier()
    if rank == 0:
        print("Distributed filesystem save finished.")
        print("Comparing DFS-saved HF weights with the source HF weights.")
        compare_saved_weights(args.model_path, distributed_save_path)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
