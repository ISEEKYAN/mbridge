# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build-only smoke test for the mbridge DeepSeek-V4 bridge.

Verifies that ``AutoBridge.from_pretrained`` selects ``DeepseekV4Bridge``,
that ``_build_config()`` produces an ``MLATransformerConfig`` carrying the
DSv4-specific fields (``experimental_attention_variant``,
``csa_compress_ratios``, ``enable_hyper_connections``, ``moe_n_hash_layers``,
``activation_func_clamp_value``, ...), and that ``get_model`` builds a
random-initialised model on which a single forward pass produces finite
output.

This test does NOT exercise:
  * Real DSv4 checkpoint import (FP8 / MXFP4 dequantisation is not
    implemented in the mbridge DSv4 bridge — use
    ``NVIDIA-NeMo/Megatron-Bridge`` for that path).
  * TP > 1 (DSv4 hybrid attention asserts TP==1 upstream in MCore).
  * CP / sequence packing (gated on upstream Megatron-LM PRs).

Usage
-----
Single GPU::

    torchrun --nproc_per_node=1 example/deepseek_v4/test_dsv4_build.py \\
        --config_path /path/to/toy_dsv4_config_dir

The directory passed to ``--config_path`` must contain a ``config.json``
with ``model_type: "deepseek_v4"``.
"""

import argparse
import json
import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import PretrainedConfig

from mbridge import AutoBridge

# ----- distributed setup ----------------------------------------------------


def init_distributed(*, tp: int = 1, pp: int = 1, cp: int = 1, ep: int = 1) -> None:
    """Initialise NCCL + Megatron model-parallel state."""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    model_parallel_cuda_manual_seed(0)


# ----- config-level field checks --------------------------------------------


_REQUIRED_DSV4_FIELDS = (
    "experimental_attention_variant",
    "csa_compress_ratios",
    "csa_window_size",
    "dsa_indexer_n_heads",
    "dsa_indexer_head_dim",
    "dsa_indexer_topk",
    "enable_hyper_connections",
    "num_residual_streams",
    "mhc_sinkhorn_iterations",
    "moe_n_hash_layers",
    "actual_vocab_size",
    "activation_func_clamp_value",
    "q_lora_rank",
    "o_groups",
    "o_lora_rank",
    "v_head_dim",
    "qk_pos_emb_head_dim",
)


def check_dsv4_fields(bridge) -> None:
    """Assert the resolved config carries the expected DSv4-specific fields."""
    config = bridge.config
    missing = [name for name in _REQUIRED_DSV4_FIELDS if not hasattr(config, name)]
    if missing:
        raise AssertionError(
            f"MLATransformerConfig missing expected DSv4 fields: {missing}. "
            "Check that the loaded MCore commit includes the DSv4 hybrid "
            "attention / mHC / hash-routing / ClampedSwiGLU patches."
        )

    if getattr(config, "experimental_attention_variant", None) != "dsv4_hybrid":
        raise AssertionError(
            "Expected experimental_attention_variant='dsv4_hybrid'; "
            f"got {getattr(config, 'experimental_attention_variant', None)!r}."
        )
    if not config.enable_hyper_connections:
        raise AssertionError("DSv4 bridge must enable hyper-connections by default.")

    expected_ratios = bridge.hf_config.num_hidden_layers + (
        getattr(bridge.hf_config, "num_nextn_predict_layers", 0) or 0
    )
    actual_ratios = len(config.csa_compress_ratios)
    if actual_ratios != expected_ratios:
        raise AssertionError(
            f"csa_compress_ratios length {actual_ratios} != "
            f"num_hidden_layers + num_nextn_predict_layers ({expected_ratios})."
        )


# ----- forward step ---------------------------------------------------------


def make_random_batch(seq_len: int, vocab_size: int, micro_batch_size: int = 1):
    """Produce a random ``input_ids`` batch for the forward smoke."""
    input_ids = torch.randint(
        0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long
    ).cuda()
    return {"input_ids": input_ids}


def mcore_fwd_fn(data_iterator, model):
    sample = next(data_iterator)
    output_tensor = model(
        input_ids=sample["input_ids"],
        position_ids=None,
        attention_mask=None,
    )
    if isinstance(output_tensor, tuple):
        output_tensor = output_tensor[0]

    def loss_fn(output_tensor, non_loss_data=True):
        loss = output_tensor.mean()
        return loss, {"loss": loss.detach()}

    return output_tensor, loss_fn


# ----- entrypoint -----------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(description="DSv4 mbridge build smoke test")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a HF config directory containing config.json with model_type=deepseek_v4",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help="Random batch sequence length for the forward smoke.",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    init_distributed(tp=args.tp, pp=args.pp, cp=args.cp, ep=args.ep)
    rank = torch.distributed.get_rank()

    if rank == 0:
        print(f"[rank0] loading bridge from {args.config_path!r}")

    # Load the HF config directly from JSON instead of going through
    # ``AutoConfig.from_pretrained``. The build smoke needs the config
    # attributes only; bypassing AutoConfig avoids requiring a Transformers
    # release that has registered ``deepseek_v4`` in its AUTO mapping (PR
    # #45643). Real DSv4 checkpoint loaders should still use
    # ``AutoBridge.from_pretrained``.
    config_json = os.path.join(args.config_path, "config.json")
    with open(config_json) as f:
        config_dict = json.load(f)
    hf_config = PretrainedConfig(**config_dict)
    hf_config.model_type = config_dict["model_type"]
    bridge = AutoBridge.from_config(hf_config)
    if rank == 0:
        print(f"[rank0] bridge class: {type(bridge).__name__}")
    assert type(bridge).__name__ == "DeepseekV4Bridge", (
        f"Expected DeepseekV4Bridge, got {type(bridge).__name__}. "
        "Is mbridge.models.deepseek_v4 importable in this environment?"
    )

    # The mbridge DSv4 bridge defaults use_fused_mhc=True (matches NeMo MB);
    # disable for the smoke so we don't depend on mHC fused kernels here.
    bridge.config.use_fused_mhc = False

    check_dsv4_fields(bridge)
    if rank == 0:
        print("[rank0] DSv4 config field check PASSED")

    model = bridge.get_model(
        weight_path=None,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    assert len(model) == 1
    model_module = model[0]
    if rank == 0:
        n_params = sum(p.numel() for p in model_module.parameters())
        print(f"[rank0] model built: {n_params:,} params")

    sample = make_random_batch(args.seq_len, bridge.hf_config.vocab_size)
    torch.distributed.barrier()
    with torch.no_grad():
        fwd_bwd = get_forward_backward_func()
        outputs = fwd_bwd(
            forward_step_func=mcore_fwd_fn,
            data_iterator=iter([sample]),
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=args.seq_len,
            decoder_seq_length=args.seq_len,
            micro_batch_size=1,
        )

    if mpu.is_pipeline_last_stage():
        loss = outputs[0]["loss"]
        finite = torch.isfinite(loss).all()
        if (
            rank == 0
            or torch.distributed.get_rank() == torch.distributed.get_world_size() - 1
        ):
            print(
                f"[rank{torch.distributed.get_rank()}] forward done. "
                f"loss={loss.item():.6f} finite={bool(finite)}"
            )
        assert bool(finite), "Forward produced non-finite loss"

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        print("[rank0] DSv4 build smoke OK")


if __name__ == "__main__":
    main()
