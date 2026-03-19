# Test script for MTP (Multi-Token Prediction) layers with Megatron
#
# This script:
#   1. Loads the full Qwen3.5 model (including MTP layers) via megatron using mbridge.
#   2. Runs a forward pass with labels to trigger the MTP code path.
#   3. Captures per-head logits (each MTP head + main head) by hooking the output_layer.
#   4. Verifies that the MTP head logits predict the correct future token.
#
# CAPTURE ORDER (how megatron calls output_layer when labels are given):
#   The gpt_model.py postprocess:
#     for k in range(mtp_num_layers):          # MTP heads first
#         compute_output_layer_and_language_model_loss(hidden_states_list[k+1], ...)
#     compute_output_layer_and_language_model_loss(hidden_states, ...)  # main head last
#
#   => logit_captures[k]  (k < mtp_num_layers) : MTP head k
#      logit_captures[-1]                       : main head
#
# SHIFT SEMANTICS:
#   - Main head logits[i] predicts token[i+1]  (standard LM)
#   - MTP  head k logits[i] predicts token[i + (k+2)]
#     Reason: mtp_labels is roll-shifted (k+1) times (once per MTP head + 1 for
#             the initial shift we provide as labels = input[1:]).
#     But our hook captures the *logits* (before the loss), so we compare logits[i]
#     against the token the model was *asked to predict at that position*.
#     For MTP head k:
#       mtp_labels after (k+1) shifts = token[i + k+1+1] = token[i + k+2]   (0-indexed)
#     So MTP head k logits[i] should be compared against input_ids[i + k + 2].
#
# Usage (MoE model, TP=2, EP=4):
#   torchrun --nproc_per_node=8 example/qwen3_5/test_mtp_logits.py \
#       --model_path /path/to/Qwen3.5-35B-A3B --tp 2 --ep 4 --etp 1
#
# Test result on Qwen3.5-35B-A3B model
# ================================================================
#   MTP HEAD 0  —  logits[i] predicts token[i+2]
#   Top-1 accuracy : 43.08%  (28/65 valid positions)
# ================================================================

#   --- Spot-check: first 12 positions ---
#   pos=  0 | pred=    314 ' of'          | gt=    314 ' of'          | ✓
#   pos=  1 | pred=    279 ' the'         | gt=   9338 ' France'      | ✗
#   pos=  2 | pred=    369 ' is'          | gt=    369 ' is'          | ✓
#   pos=  3 | pred=  11751 ' Paris'       | gt=  11751 ' Paris'       | ✓
#   pos=  4 | pred=     13 '.'            | gt=     13 '.'            | ✓
#   pos=  5 | pred=    198 '\n'           | gt=    561 ' The'         | ✗
#   pos=  6 | pred=   6511 ' capital'     | gt= 242476 ' Eiff'        | ✗
#   pos=  7 | pred=    684 'so'           | gt=    300 'el'           | ✗
#   pos=  8 | pred=  21262 ' Tower'       | gt=  21262 ' Tower'       | ✓
#   pos=  9 | pred=    369 ' is'          | gt=    557 ' was'         | ✗
#   pos= 10 | pred=   5617 ' built'       | gt=   5617 ' built'       | ✓
#   pos= 11 | pred=    303 ' in'          | gt=    303 ' in'          | ✓

# ================================================================
#   MAIN HEAD  —  logits[i] predicts token[i+1]
#   Top-1 accuracy : 68.18%  (45/66 valid positions)
# ================================================================

# ================================================================
#   SUMMARY
# ================================================================
#   [PASS] MTP head 0 (predicts token[i+2]): top-1 acc = 43.08% (28/65)
#   [PASS] Main head (predicts token[i+1]): top-1 acc = 68.18% (45/66)
# ================================================================


import argparse

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer

from mbridge import AutoBridge
from mbridge.core.util import unwrap_model


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def gather_output_from_cp(input_: torch.Tensor, seq_dim: int, cp_size: int, cp_group):
    """Gather context-parallel sequence chunks back into the full sequence."""
    assert seq_dim in [0, 1] and input_.dim() > seq_dim
    input_ = input_.view(
        *input_.shape[:seq_dim],
        2,
        input_.shape[seq_dim] // 2,
        *input_.shape[seq_dim + 1:],
    )
    gathered = [torch.zeros_like(input_) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, input_, group=cp_group)
    reordered = [None] * (2 * cp_size)
    if seq_dim == 1:
        for r in range(cp_size):
            reordered[r] = gathered[r][:, 0]
            reordered[2 * cp_size - r - 1] = gathered[r][:, 1]
    else:  # seq_dim == 0
        for r in range(cp_size):
            reordered[r] = gathered[r][0]
            reordered[2 * cp_size - r - 1] = gathered[r][1]
    return torch.cat(reordered, dim=seq_dim)


def init_distributed(tp=2, pp=1, cp=1, vpp=None, ep=1, etp=None):
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test MTP logits: load the full Qwen3.5 model (including MTP layers) "
            "and verify that the MTP head logits predict the correct future token."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp",  type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--pp",  type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp",  type=int, default=1, help="Context parallel size")
    parser.add_argument("--vpp", type=int, default=None, help="Virtual pipeline parallel size")
    parser.add_argument("--ep",  type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp", type=int, default=None, help="Expert tensor parallel size")
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "The capital of France is Paris. "
            "The Eiffel Tower was built in 1889 and stands 330 meters tall. "
            "It was designed by Gustave Eiffel as the entrance arch to the 1889 World Fair. "
            "Paris is also known as the City of Light, attracting millions of tourists each year."
        ),
        help="Input text for the MTP token-prediction test",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Top-k accuracy threshold (default 1 = exact argmax match)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Top-k token accuracy helper
# ---------------------------------------------------------------------------

def compute_top_k_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, k: int = 1):
    """Compute top-k token prediction accuracy.

    Args:
        logits:     [B, S, V]  - raw (un-normalised) logits
        target_ids: [B, S]     - ground-truth token ids; 0 = padding (ignored)
        k:          top-k window
    Returns:
        (accuracy : float, num_correct : int, num_total : int)
    """
    valid_mask = target_ids != 0  # [B, S]
    if k == 1:
        pred    = logits.argmax(dim=-1)                          # [B, S]
        correct = (pred == target_ids) & valid_mask
    else:
        topk    = logits.topk(k, dim=-1).indices                 # [B, S, k]
        correct = (topk == target_ids.unsqueeze(-1)).any(dim=-1) & valid_mask

    num_total   = valid_mask.sum().item()
    num_correct = correct.sum().item()
    acc = num_correct / num_total if num_total > 0 else 0.0
    return acc, num_correct, num_total


# ---------------------------------------------------------------------------
# Forward step function
# ---------------------------------------------------------------------------

def make_fwd_fn_with_labels(input_ids_gpu: torch.Tensor):
    """Return a forward step function that passes shifted labels to the model.

    Passing labels (not None) causes gpt_model.py to:
      1. Run the MTP block (if mtp_num_layers > 0).
      2. Call compute_output_layer_and_language_model_loss for each MTP head.
      3. Call compute_output_layer_and_language_model_loss for the main head.
    Our hook on output_layer will capture all these calls.
    """

    def fwd_fn(data_iterator, model):
        _ = next(data_iterator)  # consume iterator (input captured via closure)
        # Shift-right labels: target[i] = input[i+1]; pad last position with 0
        labels = F.pad(input_ids_gpu[:, 1:], (0, 1), value=0)  # [B, S]

        output_tensor = model(
            input_ids=input_ids_gpu,
            position_ids=None,
            attention_mask=None,
            labels=labels,
        )
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]
        assert isinstance(output_tensor, torch.Tensor)

        def loss_fn(output_tensor, **_kwargs):
            return output_tensor.mean(), {"loss": output_tensor.detach().mean()}

        return output_tensor, loss_fn

    return fwd_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    init_distributed(
        tp=args.tp, pp=args.pp, cp=args.cp, vpp=args.vpp, ep=args.ep, etp=args.etp
    )

    rank       = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if rank == 0:
        print(f"\n[rank{rank}] Loading model from: {args.model_path}")

    bridge = AutoBridge.from_pretrained(args.model_path)
    bridge.config.sequence_parallel = True
    bridge.config.mtp_loss_scaling_factor = 0.1

    if args.pp > 1:
        num_layer        = bridge.hf_config.text_config.num_hidden_layers
        first_last_layer = num_layer - (num_layer + args.pp - 1) // args.pp * (args.pp - 2)
        assert first_last_layer > 1, f"first_last_layer={first_last_layer} must be > 1"
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=first_last_layer // 2,
            num_layers_in_last_pipeline_stage=(first_last_layer + 1) // 2,
        )

    model = bridge.get_model(model_type=ModelType.encoder_or_decoder)
    # model.language_model.config.mtp_loss_scaling_factor = 0.1
    # print("model:", model)
    assert len(model) == 1
    bridge.load_weights(model, args.model_path, memory_efficient=False)

    mtp_num_layers = getattr(bridge.config, "mtp_num_layers", None) or 0

    if rank == 0:
        print(f"[rank{rank}] Model loaded.  mtp_num_layers = {mtp_num_layers}")

    if mtp_num_layers == 0:
        if rank == 0:
            print("[WARNING] mtp_num_layers == 0. No MTP layers to test. Exiting.")
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return

    # ------------------------------------------------------------------
    # Tokenise
    # ------------------------------------------------------------------
    tokenizer      = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    token_ids      = tokenizer(args.text, return_tensors="pt")["input_ids"]  # [1, S]
    real_seq_length = token_ids.shape[-1]

    if rank == 0:
        print(f"[rank{rank}] Input length : {real_seq_length} tokens")

    # Pad sequence to multiple of (tp * 2 * cp) for sequence parallelism
    seq_length_factor = args.tp
    if args.cp > 1:
        seq_length_factor *= args.cp * 2
    seq_length = real_seq_length
    if real_seq_length % seq_length_factor != 0:
        seq_length = (
            (real_seq_length + seq_length_factor - 1)
            // seq_length_factor * seq_length_factor
        )
        token_ids = F.pad(token_ids, (0, seq_length - real_seq_length), value=0)

    input_ids_gpu = token_ids.cuda()  # [1, S_padded]

    # ------------------------------------------------------------------
    # Register forward hook on output_layer to capture per-head logits.
    #
    # Megatron calls output_layer in this order when labels are given:
    #   - MTP head 0 … MTP head (mtp_num_layers-1)  [via mtp postprocess loop]
    #   - Main head                                   [via final postprocess]
    #
    # Total calls = mtp_num_layers + 1.
    # logit_captures[k]  for k < mtp_num_layers  → MTP head k logits
    # logit_captures[-1]                          → main head logits
    # ------------------------------------------------------------------

    # Navigate to the underlying language model on the last PP stage.
    # On other PP stages output_layer does not exist; we only hook if present.
    gpt_wrapper = unwrap_model(model[0])       # Qwen3_5VLModel
    lang_model  = gpt_wrapper.language_model   # Qwen3_5GPTModel (subclass of GPTModel)

    logit_captures = []

    hook_handle = None
    if hasattr(lang_model, "output_layer"):
        def output_layer_hook(_module, _input_tensors, output):
            """Capture logits: output[0] is [S, B, V_tp]."""
            logit_captures.append(output[0].detach())
            return output  # pass through unchanged

        hook_handle = lang_model.output_layer.register_forward_hook(output_layer_hook)

    torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Forward pass (with labels to trigger MTP computation)
    # ------------------------------------------------------------------
    with torch.no_grad():
        fwd_bwd_func = get_forward_backward_func()
        fwd_bwd_func(
            forward_step_func=make_fwd_fn_with_labels(input_ids_gpu),
            data_iterator=iter([{}]),   # dummy; real data captured via closure
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_length,
            decoder_seq_length=seq_length,
            micro_batch_size=1,
        )

    if hook_handle is not None:
        hook_handle.remove()

    # ------------------------------------------------------------------
    # Analyse results on the last pipeline stage
    # ------------------------------------------------------------------
    if mpu.is_pipeline_last_stage():

        n_captured     = len(logit_captures)
        expected_calls = 1 + mtp_num_layers

        if rank == 0:
            print(
                f"\n[rank{rank}] output_layer hook captured {n_captured} logit tensors "
                f"(expected {expected_calls} = 1 main + {mtp_num_layers} MTP)."
            )
            for i, t in enumerate(logit_captures):
                print(f"  logit_captures[{i}].shape = {t.shape}")

        # Safety check
        assert n_captured == expected_calls, (
            f"Expected {expected_calls} output_layer calls, got {n_captured}. "
            "Possible causes: mtp_num_layers mismatch, MTP weights not loaded, "
            "or wrong pipeline stage."
        )

        # ----------------------------------------------------------------
        # Helper: gather CP + TP dims, trim padding, return [B, S_real, V]
        # ----------------------------------------------------------------
        def gather_and_trim(logit: torch.Tensor) -> torch.Tensor:
            """logit: [S_padded, B, V_tp] → [B, S_real, V_full]."""
            if mpu.get_context_parallel_world_size() > 1:
                logit = gather_output_from_cp(
                    logit,
                    seq_dim=0,
                    cp_size=mpu.get_context_parallel_world_size(),
                    cp_group=mpu.get_context_parallel_group(),
                )
            if mpu.get_tensor_model_parallel_world_size() > 1:
                logit = gather_from_tensor_model_parallel_region(logit)
            logit = logit[:real_seq_length, :, :]        # [S_real, B, V]
            return logit.permute(1, 0, 2).contiguous()   # [B, S_real, V]

        # After gather_from_tensor_model_parallel_region the full vocab is only
        # present on the last TP rank (all ranks receive a copy, but let's use
        # the reporting rank for printing).
        is_reporting_rank = (rank == world_size - 1)

        input_ids_cpu = token_ids[:, :real_seq_length]  # [1, S_real]

        all_results = []  # (head_name, acc, n_correct, n_total)

        # ----------------------------------------------------------------
        # MTP heads:  captures[0 .. mtp_num_layers-1]
        #   MTP head k logits[i] was trained to predict token[i + k + 2].
        #   Reason: we supplied labels = input[i+1], which megatron further
        #   shifts by (k+1) inside the MTP loop → effective target is
        #   input[i + 1 + (k+1)] = input[i + k + 2].
        # ----------------------------------------------------------------
        for k in range(mtp_num_layers):
            mtp_logits = gather_and_trim(logit_captures[k])  # [B, S, V]
            mtp_logits = mtp_logits.cpu()
            shift = k + 2  # how far ahead this head predicts (0-indexed into input_ids)
            if shift < real_seq_length:
                mtp_target = F.pad(
                    input_ids_cpu[:, shift:], (0, shift), value=0
                )  # [B, S]
            else:
                mtp_target = torch.zeros_like(input_ids_cpu)

            acc, n_correct, n_total = compute_top_k_accuracy(
                mtp_logits, mtp_target, k=args.top_k
            )
            all_results.append((f"MTP head {k} (predicts token[i+{shift}])", acc, n_correct, n_total))

            if is_reporting_rank:
                print(
                    f"\n{'='*64}\n"
                    f"  MTP HEAD {k}  —  logits[i] predicts token[i+{shift}]\n"
                    f"  Top-{args.top_k} accuracy : {acc*100:.2f}%  "
                    f"({n_correct}/{n_total} valid positions)\n"
                    f"{'='*64}"
                )

                # Spot-check: show first few positions
                print(f"\n  --- Spot-check: first 12 positions ---")
                pred_ids = mtp_logits[0].argmax(dim=-1)  # [S]
                for pos in range(min(12, real_seq_length - shift)):
                    pred_tok = pred_ids[pos].item()
                    gt_tok   = mtp_target[0, pos].item()
                    match    = "✓" if pred_tok == gt_tok else "✗"
                    pred_str = tokenizer.decode([pred_tok])
                    gt_str   = tokenizer.decode([gt_tok])
                    print(
                        f"  pos={pos:3d} | "
                        f"pred={pred_tok:7d} {pred_str!r:14s} | "
                        f"gt={gt_tok:7d} {gt_str!r:14s} | {match}"
                    )

        # ----------------------------------------------------------------
        # Main head:  captures[-1]
        #   logits[i] predicts token[i+1]
        # ----------------------------------------------------------------
        main_logits = gather_and_trim(logit_captures[-1]).cpu()    # [B, S, V]
        main_target = F.pad(input_ids_cpu[:, 1:], (0, 1), value=0)  # [B, S]
        main_acc, main_correct, main_total = compute_top_k_accuracy(
            main_logits, main_target, k=args.top_k
        )
        all_results.append((f"Main head (predicts token[i+1])", main_acc, main_correct, main_total))

        if is_reporting_rank:
            print(
                f"\n{'='*64}\n"
                f"  MAIN HEAD  —  logits[i] predicts token[i+1]\n"
                f"  Top-{args.top_k} accuracy : {main_acc*100:.2f}%  "
                f"({main_correct}/{main_total} valid positions)\n"
                f"{'='*64}"
            )

        # ----------------------------------------------------------------
        # Summary & assertions
        # ----------------------------------------------------------------
        if is_reporting_rank:
            print(f"\n{'='*64}")
            print("  SUMMARY")
            print(f"{'='*64}")
            for head_name, acc, nc, nt in all_results:
                status = "PASS" if acc > 0.0 else "FAIL"
                print(f"  [{status}] {head_name}: top-{args.top_k} acc = {acc*100:.2f}% ({nc}/{nt})")
            print(f"{'='*64}\n")

            # Hard assertion: every head must have non-zero accuracy
            for head_name, acc, _, _ in all_results:
                assert acc > 0.0, (
                    f"\n[FAIL] {head_name} top-{args.top_k} accuracy = 0.0!\n"
                    "This indicates that MTP weights were not loaded correctly "
                    "or the MTP forward pass is broken."
                )
            print("[ALL TESTS PASSED] MTP logits correctly predict future tokens.\n")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
