# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 2.load_model_and_export_multiple_gpus.py --model_path /path/to/model


import argparse
import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge
from mbridge.core.util import load_one_hf_weight


def init_distributed(tp=2, pp=1, cp=1, vpp=1, ep=1):
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
    )
    model_parallel_cuda_manual_seed(0)


def main():
    # Parse command line arguments
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
        "--vpp", type=int, default=1, help="Virtual pipeline model parallel size"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert model parallel size")
    args = parser.parse_args()

    # Initialize distributed environment
    init_distributed(tp=args.tp, pp=args.pp, cp=args.cp, vpp=args.vpp, ep=args.ep)

    # Load model
    hf_model_path = args.model_path
    print(f"rank{torch.distributed.get_rank()}: start loading model")
    bridge = AutoBridge.from_pretrained(hf_model_path)
    model = bridge.get_model()
    print(
        f"rank{torch.distributed.get_rank()}: start loading weights from {hf_model_path}"
    )
    bridge.load_weights(model, hf_model_path)

    # export weights
    for k, v in bridge.export_weights(model):
        gt = load_one_hf_weight(hf_model_path, k).to(v.device)
        assert v.shape == gt.shape, f"mismatch of {k}"
        assert v.sum().item() == gt.sum().item(), f"mismatch of {k}"
        if torch.distributed.get_rank() == 1:
            print(k, "export ok")


if __name__ == "__main__":
    main()
