import torch
import triton
import triton.language as tl


@triton.jit
def compute_identity_kernel(
    top_k,
    hidden_states_ptr,
    expert_scales_ptr,
    num_tokens,
    output_ptr,
    hidden_dim,
    scales_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


# -------------------------
# mark_used_experts_kernel
# -------------------------
@triton.jit
def mark_used_experts_kernel(
    expert_indices_ptr,
    used_experts_ptr,
    num_tokens,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return

    # initialize used_experts for this token to 0
    for expert_idx_start in range(0, num_experts, BLOCK_SIZE):
        idx = expert_idx_start + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_experts
        tl.store(used_experts_ptr + token_idx * num_experts + idx, 0, mask=mask)

    # mark valid experts as used (1 = True), with bounds check
    for k in range(topk):
        expert_idx = tl.load(expert_indices_ptr + token_idx * topk + k)
        # only store if in valid range [0, num_experts)
        if (expert_idx >= 0) & (expert_idx < num_experts):
            tl.store(used_experts_ptr + token_idx * num_experts + expert_idx, 1)


# -------------------------
# reassign_invalid_kernel (no break)
# -------------------------
@triton.jit
def reassign_invalid_kernel(
    expert_indices_ptr,
    result_ptr,
    used_experts_ptr,
    num_tokens,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return

    # For each slot in topk that is -1, try to find a candidate.
    # We avoid 'break' by using an assigned flag.
    for k in range(topk):
        inp = tl.load(expert_indices_ptr + token_idx * topk + k)
        # only handle invalid slots
        if inp == -1:
            assigned = 0  # 0 -> not assigned yet, 1 -> assigned
            # Use simple offset based on token_idx and k for balanced distribution
            start_offset = ((token_idx + k) * 7) % num_experts
            # scan candidates starting from offset for better load balancing
            for offset in range(num_experts):
                candidate = (start_offset + offset) % num_experts
                # load used flag (uint8 0/1)
                is_used = tl.load(used_experts_ptr + token_idx * num_experts + candidate)
                # only if not used and not already assigned
                if (is_used == 0) & (assigned == 0):
                    # mark as used and write to result
                    tl.store(used_experts_ptr + token_idx * num_experts + candidate, 1)
                    tl.store(result_ptr + token_idx * topk + k, candidate)
                    # set assigned flag (no break; subsequent iterations will skip due to assigned==1)
                    assigned = 1
            # if after scanning all candidates still not assigned -> fallback deterministic assignment
            if assigned == 0:
                fallback = (token_idx + k) % num_experts
                tl.store(result_ptr + token_idx * topk + k, fallback)


# -------------------------
# Python wrapper helper
# -------------------------
def reassign_invalid_expert_indices_triton(
    expert_indices: torch.Tensor,
    num_experts: int,
    MAX_TOPK: int = 12,
):
    # ensure contiguous and dtypes suitable for Triton
    assert expert_indices.dim() == 2, "expected [num_tokens, topk]"
    num_tokens, topk = expert_indices.shape
    assert topk <= MAX_TOPK, f"topk ({topk}) must be <= MAX_TOPK ({MAX_TOPK})"
    assert num_experts <= 4096, "just a sanity upper bound"

    device = expert_indices.device

    # cast to int32 and contiguous
    expert_indices_i32 = expert_indices.to(torch.int32).contiguous()
    result = expert_indices_i32.clone()

    # fast path: nothing to do
    if not (result == -1).any():
        return result

    # used_experts must be uint8 (0/1)
    used_experts = torch.zeros((num_tokens, num_experts), dtype=torch.uint8, device=device)

    BLOCK_SIZE = 128
    # launch mark kernel
    mark_used_experts_kernel[(num_tokens,)](
        expert_indices_i32,
        used_experts,
        num_tokens,
        topk=topk,
        num_experts=num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # launch reassign kernel
    reassign_invalid_kernel[(num_tokens,)](
        expert_indices_i32,
        result,
        used_experts,
        num_tokens,
        topk=topk,
        num_experts=num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # result is int32 on device
    return result


def zero_experts_compute_triton(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    expert_indices = expert_indices.detach().clone()  # LongTensor, no grad anyway but safe
    expert_scales = expert_scales.detach().clone()

    N = expert_indices.numel()
    top_k = expert_indices.size(-1)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales[zero_expert_mask] = 0.0

    normal_expert_mask = expert_indices >= num_experts
    expert_indices[normal_expert_mask] = -1
    expert_scales[normal_expert_mask] = 0.0

    output = torch.zeros_like(hidden_states).to(hidden_states.device)
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)

    grid = lambda meta: (num_tokens * (hidden_dim // meta["BLOCK_SIZE"]),)
    compute_identity_kernel[grid](
        top_k,
        hidden_states,
        zero_expert_scales,
        num_tokens,
        output,
        hidden_dim,
        zero_expert_scales.stride(0),
        BLOCK_SIZE=256,
    )

    return output, expert_indices, expert_scales
