# Adapted from NVIDIA Megatron-Bridge

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te
from mbridge.peft.base import PEFT
# Import canonical split-adapter wrappers for gather/merge support.
# These are only imported here (not in canonical_lora → avoids circular deps).
from mbridge.peft.canonical_lora import (LoRALinearSplitFC1UpGate,
                                         LoRALinearSplitQKV)
from mbridge.peft.lora_layers import (LinearAdapter, LoRALinear,
                                      LoRATopKRouter, TEFusedLoRALinear,
                                      TELinearAdapter, patch_linear_module)
from mbridge.peft.module_matcher import ModuleMatcher
from mbridge.peft.utils import (ParallelLinearAdapter,
                                get_adapter_attributes_from_linear,
                                is_expert_linear)
from megatron.core import parallel_state
from megatron.core.tensor_parallel import (ColumnParallelLinear,
                                           RowParallelLinear)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

logger = logging.getLogger(__name__)

try:
    import bitsandbytes

    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRa to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
        lora_A_init_method (str): Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for the low-rank matrix B. Defaults to "zero".
        lora_dtype (torch.dtype): Parameter data type for LoRA weights. Default None (will use model's dtype).
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        # Skip already transformed modules
        adapter_types = (LinearAdapter, LoRALinear, LoRATopKRouter)
        adapter_types = adapter_types + (TELinearAdapter,)
        if isinstance(module, adapter_types):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans
            if isinstance(module, nn.Linear) or (module.__class__ == te.Linear):
                # Will use the `patch_linear_module` function if:
                # - is FSDP v1
                # - is DTensor (has _local_tensor attribute)
                # - has quant_state attribute
                if hasattr(module.weight.data, "_local_tensor") or (
                    HAVE_BNB
                    and getattr(module, "quant_state", None) is not None
                    and module.quant_state.__class__ == bitsandbytes.functional.QuantState
                ):
                    lora_cls = patch_linear_module
                elif module.__class__ == te.Linear:
                    lora_cls = TELinearAdapter
                else:
                    lora_cls = LinearAdapter

                return lora_cls(
                    module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert)

            enable_op_fuser = (
                hasattr(module, "config")
                and getattr(module.config, "use_transformer_engine_op_fuser", False)
                # TP not yet supported
                and parallel_state.get_tensor_model_parallel_world_size() == 1
            )

            logging.info(f"Adding lora to: {full_name}")
            adapter = ParallelLinearAdapter(
                attrs.in_features,
                attrs.out_features,
                self.dim,
                base_linear_name=full_name,
                activation="identity",
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(module, "config", None),
                alpha=self.alpha,
                is_expert=is_expert,
                a2a_experimental=self.a2a_experimental,
                disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )
            if isinstance(module, TopKRouter):
                return LoRATopKRouter(module, adapter)
            if enable_op_fuser:
                return TEFusedLoRALinear(module, adapter)
            else:
                return LoRALinear(module, adapter)
        return module


def _gather_parallel_weight(weight: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Gather a TP-sharded weight tensor to its full (un-sharded) size.

    ColumnParallelLinear stores weight as ``(out/TP, in)`` — gather dim 0.
    RowParallelLinear stores weight as ``(out, in/TP)`` — gather dim 1.
    """
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return weight

    tp_group = parallel_state.get_tensor_model_parallel_group()
    gathered = [torch.empty_like(weight) for _ in range(tp_size)]
    dist.all_gather(gathered, weight.contiguous(), group=tp_group)

    if isinstance(module, RowParallelLinear):
        return torch.cat(gathered, dim=1)
    else:
        return torch.cat(gathered, dim=0)


def _deinterleave_gathered_lora_b(
    gathered_b: torch.Tensor, stride: int, tp_size: int
) -> torch.Tensor:
    """Permute a gathered LoRA-B (linear_out) from interleaved to sequential layout.

    When TP > 1 and the base layer has stride > 1 (e.g. SwiGLU linear_fc1),
    each rank's B_local has rows that alternate between stride components
    (gate and up for stride=2). After naive concatenation across TP ranks,
    the layout is interleaved:

        [rank0_gate, rank0_up, rank1_gate, rank1_up, ...]

    This function permutes to sequential layout:

        [gate_all, up_all]

    which is the correct layout for HF export and for computing the full
    delta matrix during merge.
    """
    if stride <= 1 or tp_size <= 1:
        return gathered_b

    total_rows = gathered_b.shape[0]
    per_rank = total_rows // tp_size
    per_stride_per_rank = per_rank // stride

    parts = []
    for s in range(stride):
        for r in range(tp_size):
            start = r * per_rank + s * per_stride_per_rank
            end = start + per_stride_per_rank
            parts.append(gathered_b[start:end])
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Megatron-Core → HuggingFace PEFT name mapping
# ---------------------------------------------------------------------------

_MCORE_TO_HF_LORA_SUFFIX = {
    "linear_in": "lora_A",
    "linear_out": "lora_B",
}

# Mapping from CanonicalLoRA sub-adapter name → index into the bridge's
# fused weight-name list for the parent megatron module.
# e.g. bridge maps linear_qkv.weight → [q_proj.weight, k_proj.weight, v_proj.weight]
#      → adapter_q → index 0, adapter_k → index 1, adapter_v → index 2.
_CANONICAL_ADAPTER_TO_HF_INDEX = {
    "adapter_q": 0,
    "adapter_k": 1,
    "adapter_v": 2,
    "adapter_gate": 0,
    "adapter_up": 1,
}


def _combine_hf_module_names(hf_weight_names: List[str]) -> str:
    """Derive a single fused adapter module path from multiple HF weight names.

    For fused MCore layers (e.g. ``linear_qkv`` → q/k/v, ``linear_fc1`` →
    gate/up), the bridge returns multiple HF weight names.  This function
    combines them into a single name suitable for the adapter key.

    Examples::

        ["model.layers.0.self_attn.q_proj.weight",
         "model.layers.0.self_attn.k_proj.weight",
         "model.layers.0.self_attn.v_proj.weight"]
        → "model.layers.0.self_attn.qkv_proj"

        ["model.layers.0.mlp.gate_proj.weight",
         "model.layers.0.mlp.up_proj.weight"]
        → "model.layers.0.mlp.gate_up_proj"
    """
    import os

    bases = [n.rsplit(".", 1)[0] for n in hf_weight_names]

    common_prefix = os.path.commonprefix(bases)
    if common_prefix and not common_prefix.endswith("."):
        common_prefix = common_prefix[: common_prefix.rfind(".") + 1]

    suffixes = [b[len(common_prefix):] for b in bases]

    reversed_suffixes = [s[::-1] for s in suffixes]
    common_suffix = os.path.commonprefix(reversed_suffixes)[::-1]

    strip_len = len(common_suffix)
    unique_parts = [s[:-strip_len] if strip_len else s for s in suffixes]

    if all(len(p) <= 1 for p in unique_parts):
        combined = "".join(unique_parts) + common_suffix
    else:
        combined = "_".join(unique_parts) + common_suffix

    return common_prefix + combined


def mcore_adapter_name_to_hf(mcore_name: str, bridge=None) -> str:
    """Convert a Megatron-Core adapter parameter name to HF PEFT format.

    When *bridge* is provided the mapping is derived dynamically via
    ``bridge._weight_name_mapping_mcore_to_hf``, which handles every
    model architecture the bridge supports.  Without a bridge the
    function is a no-op passthrough (the name is prefixed with
    ``base_model.model.`` only).

    Supports both standard LoRA (single adapter per fused layer) and
    CanonicalLoRA (multiple sub-adapters per fused layer, e.g.
    ``adapter.adapter_q.linear_in.weight``).

    Parameters
    ----------
    mcore_name : str
        Full Megatron-Core adapter parameter name, e.g.
        ``decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight``
        or ``decoder.layers.0.self_attention.linear_qkv.adapter.adapter_q.linear_in.weight``
    bridge : optional
        An mbridge ``Bridge`` instance.

    Returns
    -------
    str
        HF PEFT parameter name, e.g.
        ``base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight``
    """
    import re

    # --- CanonicalLoRA nested adapter path ---
    # e.g. …linear_qkv.adapter.adapter_q.linear_in.weight
    m_canonical = re.match(
        r"(.+)\.adapter\.(adapter_\w+)\.linear_(in|out)\.weight$",
        mcore_name,
    )
    if m_canonical is not None:
        base_module_path = m_canonical.group(1)
        sub_adapter = m_canonical.group(2)      # e.g. "adapter_q"
        adapter_type = m_canonical.group(3)      # "in" or "out"
        lora_suffix = _MCORE_TO_HF_LORA_SUFFIX[f"linear_{adapter_type}"]

        if bridge is not None:
            mcore_weight_name = f"{base_module_path}.weight"
            hf_names = bridge._weight_name_mapping_mcore_to_hf(mcore_weight_name)
            # Select the correct HF name for this sub-adapter
            idx = _CANONICAL_ADAPTER_TO_HF_INDEX.get(sub_adapter, 0)
            if idx < len(hf_names):
                hf_base = hf_names[idx].rsplit(".", 1)[0]
            elif len(hf_names) == 1:
                hf_base = hf_names[0].rsplit(".", 1)[0]
            else:
                # Fallback: use the fused name with sub-adapter suffix
                fused_base = _combine_hf_module_names(hf_names)
                hf_base = f"{fused_base}_{sub_adapter}"
            return f"base_model.model.{hf_base}.{lora_suffix}.weight"

        return f"base_model.model.{base_module_path}.{sub_adapter}.{lora_suffix}.weight"

    # --- Standard LoRA adapter path ---
    # e.g. …linear_qkv.adapter.linear_in.weight
    m = re.match(
        r"(.+)\.(adapter\.linear_(in|out)\.weight)$",
        mcore_name,
    )
    if m is None:
        return f"base_model.model.{mcore_name}"

    base_module_path = m.group(1)
    adapter_type = m.group(3)  # "in" or "out"
    lora_suffix = _MCORE_TO_HF_LORA_SUFFIX[f"linear_{adapter_type}"]

    if bridge is not None:
        mcore_weight_name = f"{base_module_path}.weight"
        hf_names = bridge._weight_name_mapping_mcore_to_hf(mcore_weight_name)

        if len(hf_names) == 1:
            hf_base = hf_names[0].rsplit(".", 1)[0]
        else:
            hf_base = _combine_hf_module_names(hf_names)

        return f"base_model.model.{hf_base}.{lora_suffix}.weight"

    return f"base_model.model.{base_module_path}.{lora_suffix}.weight"


def infer_hf_target_modules(adapter_state: Dict[str, torch.Tensor]) -> list:
    """Infer HF ``target_modules`` from adapter weight names.

    Keys look like ``...layers.0.self_attn.qkv_proj.lora_A.weight``.
    The module name (``qkv_proj``) is 3 dots from the end.
    """
    modules = set()
    for key in adapter_state:
        parts = key.rsplit(".", 3)
        if len(parts) >= 4:
            modules.add(parts[-3])
    return sorted(modules)


@torch.no_grad()
def gather_lora_state_dict(models, bridge=None) -> Dict[str, torch.Tensor]:
    """Gather full (un-sharded) LoRA adapter weights in HF PEFT format.

    When TP > 1, the adapter's ``linear_in`` and ``linear_out`` are
    parallel linear layers whose weights are sharded across TP ranks.
    This function performs ``all_gather`` to reconstruct the full tensors
    and converts parameter names to HF PEFT convention.

    Supports both standard LoRA (``LoRALinear``) and canonical LoRA
    (``LoRALinearSplitQKV``, ``LoRALinearSplitFC1UpGate``).

    Parameters
    ----------
    models : list[nn.Module]
        Unwrapped model chunks (as returned by ``unwrap_model``).
    bridge : optional
        An mbridge ``Bridge`` instance.  When provided the mcore → HF name
        mapping is derived dynamically from the bridge, supporting any model
        architecture.  When *None*, adapter names are passed through with a
        ``base_model.model.`` prefix.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from HF PEFT parameter names
        (e.g. ``base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight``)
        to full-size tensors on CPU.
    """
    adapter_state: Dict[str, torch.Tensor] = {}

    for model_chunk in models:
        for name, module in model_chunk.named_modules():
            # --- CanonicalLoRA split adapters (Q/K/V, gate/up) ---
            if isinstance(module, (LoRALinearSplitQKV, LoRALinearSplitFC1UpGate)):
                adapters_dict = module.adapter  # ModuleDict
                for sub_name, sub_adapter in adapters_dict.items():
                    if sub_adapter is None:
                        continue
                    lin_in_w = _gather_parallel_weight(
                        sub_adapter.linear_in.weight.data, sub_adapter.linear_in,
                    )
                    hf_key = mcore_adapter_name_to_hf(
                        f"{name}.adapter.{sub_name}.linear_in.weight", bridge=bridge,
                    )
                    adapter_state[hf_key] = lin_in_w.cpu()

                    lin_out_w = _gather_parallel_weight(
                        sub_adapter.linear_out.weight.data, sub_adapter.linear_out,
                    )
                    hf_key = mcore_adapter_name_to_hf(
                        f"{name}.adapter.{sub_name}.linear_out.weight", bridge=bridge,
                    )
                    adapter_state[hf_key] = lin_out_w.cpu()

                continue

            # --- Standard LoRA ---
            if not isinstance(module, LoRALinear):
                continue
            adapter = module.adapter

            lin_in_w = _gather_parallel_weight(
                adapter.linear_in.weight.data, adapter.linear_in,
            )
            hf_key = mcore_adapter_name_to_hf(
                f"{name}.adapter.linear_in.weight", bridge=bridge,
            )
            adapter_state[hf_key] = lin_in_w.cpu()

            lin_out_w = _gather_parallel_weight(
                adapter.linear_out.weight.data, adapter.linear_out,
            )
            # For strided base layers (e.g. SwiGLU FC1), the gathered B has
            # interleaved layout; permute to sequential for correct HF export.
            stride = getattr(module.to_wrap, 'stride', 1)
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if stride > 1 and tp_size > 1:
                lin_out_w = _deinterleave_gathered_lora_b(
                    lin_out_w, stride, tp_size,
                )
            hf_key = mcore_adapter_name_to_hf(
                f"{name}.adapter.linear_out.weight", bridge=bridge,
            )
            adapter_state[hf_key] = lin_out_w.cpu()

    return adapter_state


class LoRAMerge(PEFT):
    """
    Implements the LoRA weight merge for parameter-efficient fine-tuning.
    """

    @staticmethod
    def _compute_sub_delta(linear_out, linear_in, alpha, dim, base_device):
        """Compute the full (un-sharded) LoRA delta for a single sub-adapter.

        Gathers TP-sharded weights if TP > 1.
        """
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_group = parallel_state.get_tensor_model_parallel_group()

        lin_in_w = linear_in.weight.data.to(base_device)
        lin_out_w = linear_out.weight.data.to(base_device)

        if tp_size == 1:
            return (alpha / dim) * (lin_out_w @ lin_in_w)

        # Gather linear_in along dim 0 (ColumnParallel case)
        lin_in_list = [torch.empty_like(lin_in_w) for _ in range(tp_size)]
        dist.all_gather(lin_in_list, lin_in_w.contiguous(), group=tp_group)
        lin_in_full = torch.cat(lin_in_list, dim=0)

        # Gather linear_out along dim 0
        lin_out_list = [torch.empty_like(lin_out_w) for _ in range(tp_size)]
        dist.all_gather(lin_out_list, lin_out_w.contiguous(), group=tp_group)
        lin_out_full = torch.cat(lin_out_list, dim=0)

        return (alpha / dim) * (lin_out_full @ lin_in_full)

    @staticmethod
    def _interleave_qkv_full_delta(q_delta, k_delta, v_delta, config):
        """Interleave Q, K, V full deltas into Megatron QKV packed weight order.

        The fused QKV layout (from Megatron) is:
          for each head group i:
            [Q_heads_per_group, K_1_head, V_1_head]
        """
        head_num = config.num_attention_heads
        num_query_groups = config.num_query_groups
        head_size = config.kv_channels
        heads_per_group = head_num // num_query_groups

        q_reshaped = q_delta.reshape(head_num, head_size, -1)
        k_reshaped = k_delta.reshape(num_query_groups, head_size, -1)
        v_reshaped = v_delta.reshape(num_query_groups, head_size, -1)

        interleaved_parts = []
        for g in range(num_query_groups):
            q_group = q_reshaped[g * heads_per_group: (g + 1) * heads_per_group]
            k_group = k_reshaped[g: g + 1]
            v_group = v_reshaped[g: g + 1]
            interleaved_parts.append(q_group.reshape(-1, q_delta.shape[1]))
            interleaved_parts.append(k_group.reshape(-1, q_delta.shape[1]))
            interleaved_parts.append(v_group.reshape(-1, q_delta.shape[1]))

        return torch.cat(interleaved_parts, dim=0)

    def merge(
        self,
        base_weight: torch.Tensor,
        linear_out: torch.Tensor,
        linear_in: torch.Tensor,
        alpha: int,
        dim: int,
        stride: int = 1,
    ) -> torch.Tensor:
        """
        Merges the LoRA adapter weights with the base model weights.
        Handles tensor parallelism by gathering sharded dimensions.

        For ColumnParallelLinear (e.g., linear_qkv, linear_fc1):
            - base_weight: (out_features/TP, in_features)
            - linear_in: (dim/TP, in_features) <- Need to gather this
            - linear_out: (out_features/TP, dim)
            - Target: (out_features/TP, dim) @ (dim, in_features) = (out_features/TP, in_features)

        For RowParallelLinear (e.g., linear_proj, linear_fc2):
            - base_weight: (out_features, in_features/TP)
            - linear_in: (dim, in_features/TP)
            - linear_out: (out_features/TP, dim) <- Need to gather this
            - Target: (out_features, dim) @ (dim, in_features/TP) = (out_features, in_features/TP)

        For strided ColumnParallelLinear (gated MLP linear_fc1 with stride > 1):
            The base weight has an interleaved layout across TP ranks (due to stride).
            The adapter's linear_out is a *non-strided* ColumnParallelLinear, so its
            TP sharding is a simple contiguous chunk — which does NOT match the
            interleaved layout of the base weight. This function handles this by
            gathering both linear_in and linear_out, computing the full delta, and
            then interleaving the delta chunks to match the base weight's layout.

        Args:
            base_weight (torch.Tensor): The base model weights.
            linear_out (torch.Tensor): LoRA's B matrix.
            linear_in (torch.Tensor): LoRA's A matrix.
            alpha (int): Weighting factor for the low-rank projection.
            dim (int): Dimension of the low-rank projection space.
            stride (int): Stride of the base ColumnParallelLinear (default: 1).
                          Use stride=2 for gated MLP (linear_fc1 with GLU).

        Returns:
            torch.Tensor: The merged weights.
        """

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        if tp_size == 1:
            # No tensor parallelism, simple multiplication
            lora_weight = alpha / dim * (linear_out @ linear_in)
            return base_weight + lora_weight

        tp_group = parallel_state.get_tensor_model_parallel_group()

        # Case 1: ColumnParallelLinear - linear_in is sharded on dim 0
        # linear_in: (dim/TP, in_features), linear_out: (out_features/TP, dim)
        if linear_in.shape[0] * tp_size == dim and linear_out.shape[1] == dim:
            # Gather linear_in along dimension 0 to get full dim
            linear_in_list = [torch.empty_like(linear_in) for _ in range(tp_size)]
            dist.all_gather(linear_in_list, linear_in, group=tp_group)
            linear_in_full = torch.cat(linear_in_list, dim=0)

            # adapter linear_out is non-strided ColumnParallel (contiguous chunk);
            # base weight may be strided (interleaved). For stride>1, we need to
            # gather linear_out fully and interleave the delta for this TP rank.
            if stride > 1:
                # Gather linear_out across TP to get the full B matrix
                linear_out_list = [torch.empty_like(linear_out) for _ in range(tp_size)]
                dist.all_gather(linear_out_list, linear_out, group=tp_group)
                linear_out_full = torch.cat(linear_out_list, dim=0)

                # The gathered B has interleaved layout because each rank's local
                # B contains rows for ALL stride components (gate+up).  Permute to
                # sequential [gate_all, up_all] before computing the full delta.
                linear_out_full = _deinterleave_gathered_lora_b(
                    linear_out_full, stride, tp_size,
                )

                # Full delta in sequential layout: [gate_delta, up_delta]
                full_delta = alpha / dim * (linear_out_full @ linear_in_full)
                out_features = full_delta.shape[0]

                # Split full_delta into stride parts (now correctly sequential)
                stride_chunks = full_delta.chunk(stride, dim=0)

                # Each stride chunk is further split across TP ranks
                tp_chunks_per_stride = [c.chunk(tp_size, dim=0) for c in stride_chunks]

                # For strided layout, this rank takes the tp_rank-th chunk from
                # each stride part and concatenates them
                lora_weight = torch.cat(
                    [chunks[tp_rank] for chunks in tp_chunks_per_stride],
                    dim=0,
                )
            else:
                # Non-strided: simple (out_features/TP, dim) @ (dim, in_features)
                lora_weight = alpha / dim * (linear_out @ linear_in_full)

        # Case 2: RowParallelLinear - linear_out is sharded on dim 0
        # linear_in: (dim, in_features/TP), linear_out: (out_features/TP, dim)
        elif linear_out.shape[0] * tp_size == base_weight.shape[0]:
            # Gather linear_out along dimension 0 to get full out_features
            linear_out_list = [torch.empty_like(linear_out) for _ in range(tp_size)]
            dist.all_gather(linear_out_list, linear_out, group=tp_group)
            linear_out_full = torch.cat(linear_out_list, dim=0)

            # Multiply: (out_features, dim) @ (dim, in_features/TP)
            lora_weight = alpha / dim * (linear_out_full @ linear_in)

        else:
            # Fallback: no gathering needed or already full-size
            lora_weight = alpha / dim * (linear_out @ linear_in)

        return base_weight + lora_weight

    @torch.no_grad()
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Merges the LoRA adapter with the base model weights.

        Supports standard LoRA (``LoRALinear``) and canonical LoRA
        (``LoRALinearSplitQKV``, ``LoRALinearSplitFC1UpGate``).

        Args:
            m (nn.Module): The module to apply LoRA merge to.
            name (str, optional): Name of the module to merge. Defaults to None.
            prefix (str, optional): Prefix for the module name. Defaults to None.

        Returns:
            nn.Module: The modified module with the LoRA adapter merged into the base model weights.
        """
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        # --- CanonicalLoRA: LoRALinearSplitQKV ---
        if isinstance(module, LoRALinearSplitQKV):
            base_device = module.to_wrap.weight.device
            config = module.to_wrap.config

            q_delta = k_delta = v_delta = None
            if module.adapter.adapter_q is not None:
                q_delta = self._compute_sub_delta(
                    module.adapter.adapter_q.linear_out, module.adapter.adapter_q.linear_in,
                    module.adapter.adapter_q.alpha, module.adapter.adapter_q.dim, base_device,
                )
            if module.adapter.adapter_k is not None:
                k_delta = self._compute_sub_delta(
                    module.adapter.adapter_k.linear_out, module.adapter.adapter_k.linear_in,
                    module.adapter.adapter_k.alpha, module.adapter.adapter_k.dim, base_device,
                )
            if module.adapter.adapter_v is not None:
                v_delta = self._compute_sub_delta(
                    module.adapter.adapter_v.linear_out, module.adapter.adapter_v.linear_in,
                    module.adapter.adapter_v.alpha, module.adapter.adapter_v.dim, base_device,
                )

            # Interleave into fused Megatron QKV layout
            if q_delta is not None and k_delta is not None and v_delta is not None:
                full_qkv_delta = self._interleave_qkv_full_delta(q_delta, k_delta, v_delta, config)
            else:
                # Fallback: simple concatenation Q→K→V
                parts = [d for d in [q_delta, k_delta, v_delta] if d is not None]
                full_qkv_delta = torch.cat(parts, dim=0)

            # Take TP-rank's contiguous shard of the full fused delta
            total_rows = full_qkv_delta.shape[0]
            per_rank = total_rows // tp_size if tp_size > 1 else total_rows
            start = tp_rank * per_rank
            per_rank_delta = full_qkv_delta[start:start + per_rank]

            module.to_wrap.weight.data = module.to_wrap.weight.data + per_rank_delta.to(base_device)
            return module

        # --- CanonicalLoRA: LoRALinearSplitFC1UpGate ---
        if isinstance(module, LoRALinearSplitFC1UpGate):
            base_device = module.to_wrap.weight.device
            stride = getattr(module.to_wrap, 'stride', 1)

            gate_delta = up_delta = None
            if module.adapter.adapter_gate is not None:
                gate_delta = self._compute_sub_delta(
                    module.adapter.adapter_gate.linear_out, module.adapter.adapter_gate.linear_in,
                    module.adapter.adapter_gate.alpha, module.adapter.adapter_gate.dim, base_device,
                )
            if module.adapter.adapter_up is not None:
                up_delta = self._compute_sub_delta(
                    module.adapter.adapter_up.linear_out, module.adapter.adapter_up.linear_in,
                    module.adapter.adapter_up.alpha, module.adapter.adapter_up.dim, base_device,
                )

            # Stack gate + up → (2*ffn_hidden_size, in_features)
            parts = [d for d in [gate_delta, up_delta] if d is not None]
            full_fc1_delta = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

            if tp_size > 1 and stride > 1:
                # Apply stride interleaving for the fused gate/up layout
                stride_chunks = full_fc1_delta.chunk(stride, dim=0)
                tp_chunks_per_stride = [c.chunk(tp_size, dim=0) for c in stride_chunks]
                per_rank_delta = torch.cat(
                    [chunks[tp_rank] for chunks in tp_chunks_per_stride], dim=0,
                )
            else:
                total_rows = full_fc1_delta.shape[0]
                per_rank = total_rows // tp_size if tp_size > 1 else total_rows
                start = tp_rank * per_rank
                per_rank_delta = full_fc1_delta[start:start + per_rank]

            module.to_wrap.weight.data = module.to_wrap.weight.data + per_rank_delta.to(base_device)
            return module

        # --- Standard LoRA ---
        if not isinstance(module, LoRALinear):
            return module

        # Detect stride for strided ColumnParallelLinear (gated MLP)
        stride = getattr(module.to_wrap, 'stride', 1)

        if hasattr(module.to_wrap, "weight"):
            base_device = module.to_wrap.weight.device
            merged_weight = self.merge(
                module.to_wrap.weight,
                module.adapter.linear_out.weight.to(base_device),
                module.adapter.linear_in.weight.to(base_device),
                module.adapter.alpha,
                module.adapter.dim,
                stride=stride,
            )
            module.to_wrap.weight.data = merged_weight
        else:  # TE Grouped Linear
            for i in range(module.to_wrap.num_gemms):
                base_device = getattr(module.to_wrap, f"weight{i}").device
                merged_weight = self.merge(
                    getattr(module.to_wrap, f"weight{i}"),
                    module.adapter.linear_out.weight.to(base_device),
                    module.adapter.linear_in.weight.to(base_device),
                    module.adapter.alpha,
                    module.adapter.dim,
                    stride=stride,
                )
                getattr(module.to_wrap, f"weight{i}").data = merged_weight
        return module


@contextmanager
@torch.no_grad()
def lora_merged(models):
    """Context manager that temporarily merges LoRA into base weights.

    On enter: for each LoRA-wrapped module (``LoRALinear``,
    ``LoRALinearSplitQKV``, ``LoRALinearSplitFC1UpGate``):
    (1) clones the base weight, (2) merges the LoRA delta in-place,
    and (3) swaps the wrapper with its ``to_wrap`` in the parent so that
    ``named_parameters()`` yields clean names (no ``.to_wrap.`` prefix).
    On exit, everything is restored exactly (no floating-point drift).

    Parameters
    ----------
    models : list[nn.Module]
        Unwrapped model chunks (as returned by ``unwrap_model``).
    """
    merger = LoRAMerge()
    weight_backups = []
    module_swaps = []

    _ADAPTER_WRAPPER_TYPES = (LoRALinear, LoRALinearSplitQKV, LoRALinearSplitFC1UpGate)

    for model_chunk in models:
        # Collect (parent, attr_name, lora_module) before modifying structure
        swap_list = []
        all_modules = dict(model_chunk.named_modules())
        for name, module in all_modules.items():
            if not isinstance(module, _ADAPTER_WRAPPER_TYPES):
                continue
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = all_modules[parent_name]
            else:
                parent = model_chunk
                attr_name = parts[0]
            swap_list.append((parent, attr_name, module))

        for parent, attr_name, lora_module in swap_list:
            # Backup the original base weight before merging
            if hasattr(lora_module.to_wrap, "weight"):
                orig = lora_module.to_wrap.weight.data.clone()
                weight_backups.append((lora_module.to_wrap, "weight", orig))
            else:
                # TE Grouped Linear: backup all weights
                for i in range(lora_module.to_wrap.num_gemms):
                    attr = f"weight{i}"
                    w = getattr(lora_module.to_wrap, attr)
                    orig = w.data.clone()
                    weight_backups.append((lora_module.to_wrap, attr, orig))

            # Merge using LoRAMerge.transform() which handles all adapter types
            merger.transform(lora_module)

            # Replace wrapper with to_wrap in parent so parameter names are clean
            setattr(parent, attr_name, lora_module.to_wrap)
            module_swaps.append((parent, attr_name, lora_module))

    try:
        yield
    finally:
        # Restore wrapper modules in parent
        for parent, attr_name, lora_module in module_swaps:
            setattr(parent, attr_name, lora_module)
        # Restore original weight data
        for owner, attr, orig_data in weight_backups:
            getattr(owner, attr).data = orig_data
