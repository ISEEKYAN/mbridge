"""Verify MTP sp_input_ids preparation produces correct rolled token IDs.

Core invariant being tested
---------------------------
After ``split_data_cp_rank`` (CP split) followed by ``roll_tensor(shifts=-1)``,
each position's value must equal the *next* token in the original sequence.
The last token of the full sequence should become 0.

This test catches the TP-boundary bug where pre-scattering input_ids by TP
*before* rolling causes ``roll_tensor`` to wrap within each TP chunk instead
of referencing the correct cross-TP next token.  ``roll_tensor`` only handles
CP boundaries (via isend/irecv), not TP boundaries.

Two methods are compared:
  Method A (correct):  CP-split only  → roll → check
  Method B (buggy):    TP-scatter + CP-split → roll → check

Run
---
  torchrun --nproc_per_node=4 example/qwen3_5/test_mtp_roll_correctness.py

  (TP=2, CP=2 → 4 GPUs minimum)

  Optional flags:
    --tp 2 --cp 2        (default)
    --seq_len 32         (must be divisible by tp * cp * 2)
"""

import argparse

import torch
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

# roll_tensor lives in MTP module
from megatron.core.transformer.multi_token_prediction import roll_tensor

from mbridge.core.util import split_data_cp_rank


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--cp", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=32)
    return p.parse_args()


def init_distributed(tp, cp):
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
    )
    model_parallel_cuda_manual_seed(0)


def gather_from_cp(tensor, seq_dim, cp_size, cp_group):
    """Inverse of split_data_cp_rank: all-gather and reorder bidirectional CP chunks."""
    chunk_size = tensor.shape[seq_dim] // 2
    tensor = tensor.view(
        *tensor.shape[:seq_dim],
        2,
        chunk_size,
        *tensor.shape[seq_dim + 1 :],
    )
    gathered = [torch.zeros_like(tensor) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, tensor, group=cp_group)

    reordered = [None] * (2 * cp_size)
    for r in range(cp_size):
        reordered[r] = gathered[r].select(seq_dim, 0)
        reordered[2 * cp_size - r - 1] = gathered[r].select(seq_dim, 1)
    return torch.cat(reordered, dim=seq_dim)


def build_reference(seq_len: int) -> torch.Tensor:
    """Reference rolled sequence: [1, 2, ..., S-1, 0]."""
    ref = torch.arange(1, seq_len + 1, device="cuda").unsqueeze(0)
    ref[0, -1] = 0
    return ref


def test_method_a(input_ids, cp_size, cp_group):
    """Method A (correct): CP-split only → roll → gather → compare.

    Returns
    -------
    (rolled_full, ok) : (Tensor[1, S], bool)
    """
    sp = input_ids  # [1, S], replicated across TP ranks
    if cp_size > 1:
        sp = sp.permute(1, 0).contiguous()
        sp = split_data_cp_rank(sp, cp_size, seq_dim=0)
        sp = sp.permute(1, 0).contiguous()

    rolled, _ = roll_tensor(sp, shifts=-1, dims=-1, cp_group=cp_group)

    # Gather back across CP
    full = rolled.permute(1, 0).contiguous()
    if cp_size > 1:
        full = gather_from_cp(full, seq_dim=0, cp_size=cp_size, cp_group=cp_group)
    full = full.permute(1, 0).contiguous()

    ref = build_reference(input_ids.shape[1])
    ok = torch.equal(full, ref)
    return full, ok


def test_method_b(input_ids, cp_size, cp_group, tp_group):
    """Method B (buggy): TP-scatter + CP-split → roll → gather → compare.

    Returns
    -------
    (rolled_full, ok) : (Tensor[1, S], bool)
    """
    tp_size = tp_group.size()

    sp = input_ids.permute(1, 0).contiguous()  # [S, 1]
    sp = tensor_parallel.scatter_to_sequence_parallel_region(sp)  # [S/TP, 1]
    if cp_size > 1:
        sp = split_data_cp_rank(sp, cp_size, seq_dim=0)  # [S/(TP*CP*2), 1]
    sp = sp.permute(1, 0).contiguous()  # [1, S/(TP*CP*2)]

    rolled, _ = roll_tensor(sp, shifts=-1, dims=-1, cp_group=cp_group)

    # Gather back across CP
    full = rolled.permute(1, 0).contiguous()
    if cp_size > 1:
        full = gather_from_cp(full, seq_dim=0, cp_size=cp_size, cp_group=cp_group)

    # Gather back across TP
    chunks = [torch.zeros_like(full) for _ in range(tp_size)]
    torch.distributed.all_gather(chunks, full, group=tp_group)
    full = torch.cat(chunks, dim=0)  # [S, 1]
    full = full.permute(1, 0).contiguous()

    ref = build_reference(input_ids.shape[1])
    ok = torch.equal(full, ref)
    return full, ok


def main():
    args = get_args()
    init_distributed(tp=args.tp, cp=args.cp)

    rank = torch.distributed.get_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_group = mpu.get_context_parallel_group()
    tp_group = mpu.get_tensor_model_parallel_group()

    S = args.seq_len
    assert (
        S % (tp_size * cp_size * 2) == 0
    ), f"seq_len={S} must be divisible by tp*cp*2={tp_size * cp_size * 2}"

    input_ids = torch.arange(S, device="cuda").unsqueeze(0)  # [1, S]

    # --- Test Method A ---
    full_a, ok_a = test_method_a(input_ids, cp_size, cp_group)
    # --- Test Method B ---
    full_b, ok_b = test_method_b(input_ids, cp_size, cp_group, tp_group)

    if rank == 0:
        ref = build_reference(S)
        print(f"\nConfig: TP={tp_size}, CP={cp_size}, S={S}")
        print(f"{'=' * 64}")
        print(f"Reference (correct next-token IDs):")
        print(f"  {ref[0, :16].tolist()} ...")
        print()

        print(f"Method A (CP-split only, do_scatter=True):")
        print(f"  {full_a[0, :16].tolist()} ...")
        print(f"  Correct: {ok_a}")
        print()

        print(f"Method B (TP-scatter + CP-split, do_scatter=False):")
        print(f"  {full_b[0, :16].tolist()} ...")
        print(f"  Correct: {ok_b}")

        # Find mismatched positions in Method B
        mismatches = (full_b != ref).nonzero(as_tuple=True)
        if len(mismatches[1]) > 0:
            print(f"\n  Method B errors at positions: {mismatches[1].tolist()}")
            for pos in mismatches[1].tolist():
                print(
                    f"    pos={pos}: got {full_b[0, pos].item()}, "
                    f"expected {ref[0, pos].item()}"
                )

        print(f"\n{'=' * 64}")
        assert ok_a, "[FAIL] Method A should be correct but isn't!"
        assert not ok_b, (
            "[UNEXPECTED] Method B passed — expected TP-boundary errors. "
            "This test assumes TP > 1."
        )
        print(
            "[PASS] Method A is correct, Method B has TP-boundary errors as expected."
        )
        print(f"{'=' * 64}\n")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
