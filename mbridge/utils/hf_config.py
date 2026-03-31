# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates

import importlib.metadata
import re
from functools import lru_cache
from typing import Optional

from packaging import version
from transformers import PretrainedConfig

def get_hf_rope_theta(hf_config: PretrainedConfig) -> float:
    """Return RoPE base frequency theta.

    Allow input as hf_config, hf_config with text_config attribute, or hf_config.text_config.

    Most configs expose ``rope_theta`` on the root. Newer models (e.g. Qwen3 in transformers>=5) store it under
    ``rope_parameters["rope_theta"]``, optionally nested per attention pattern when ``rope_parameters`` maps names
    to parameter dicts.
    """
    # For transformers <= 4.57.6
    if hasattr(hf_config, "rope_theta"):
        return hf_config.rope_theta
    if hasattr(hf_config, "text_config") and hasattr(hf_config.text_config, "rope_theta"):
        return hf_config.text_config.rope_theta

    # For transformers >= 5.0.0, check rope_parameters dict (optionally nested) for rope_theta
    rp = None
    if hasattr(hf_config, "rope_parameters"):
        rp = hf_config.rope_parameters
    elif hasattr(hf_config, "text_config") and hasattr(hf_config.text_config, "rope_parameters"):
        rp = hf_config.text_config.rope_parameters
    if isinstance(rp, dict):
        if "rope_theta" in rp:
            return rp["rope_theta"]
        for v in rp.values():
            if isinstance(v, dict) and "rope_theta" in v:
                return v["rope_theta"]
    raise AttributeError(
        f"{type(hf_config).__name__} has no rope_theta and no rope_parameters['rope_theta'] — "
        "cannot determine RoPE base."
    )


def get_hf_rope_theta_from_attribute(hf_config: PretrainedConfig) -> str:
    """Return the attribute name of RoPE theta.

    The hf_config must have rope_theta/rope_parameters attribute, no config subclass
    """
    if hasattr(hf_config, "rope_theta"):
        return "rope_theta"
    if hasattr(hf_config, "rope_parameters"):
        return "rope_parameters['rope_theta']"
    raise AttributeError(
        f"{type(hf_config).__name__} has no rope_theta and no rope_parameters['rope_theta'] — "
        "cannot determine RoPE base."
    )

@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check


def hf_moe_stacked_layout_default_from_transformers_version() -> bool:
    """Version-based default: stacked/fused MoE layout for current transformers version.

    In that layout, ``gate_up_proj`` has shape ``(num_experts, 2 * ffn_dim, hidden)`` and
    ``down_proj`` has shape ``(num_experts, hidden, ffn_dim)`` instead of per-expert modules.
    """
    return is_transformers_version_in_range("5.0.0", None)


# Any MoE layer index (not only 0): first layers may be dense while MoE starts later.
_MOE_PER_EXPERT_GATE_RE = re.compile(r"mlp\.experts\.\d+\.gate_proj\.weight")


def hf_moe_stacked_layout_from_checkpoint_keys(
    index_keys: Optional[set[str]],
) -> bool:
    """Infer MoE layout from **on-disk** safetensors keys (used during :meth:`Bridge.load_weights`).

    **Legacy (experts.{i} API):** per-expert tensors such as
    ``…mlp.experts.{i}.gate_proj.weight``, ``up_proj.weight``, ``down_proj.weight``.

    **Current / transformers ≥5 style:** fused tensors
    ``…mlp.experts.gate_up_proj`` (and optional ``.weight``) and ``…mlp.experts.down_proj``.

    Scans **all** index keys for these substrings/patterns so layer 0 need not be MoE
    (e.g. dense bottom layers). If neither pattern appears, falls back to
    :func:`hf_moe_stacked_layout_default_from_transformers`.
    """
    if index_keys:
        needle = "mlp.experts.gate_up_proj"
        for k in index_keys:
            if needle in k:
                return True
        for k in index_keys:
            if _MOE_PER_EXPERT_GATE_RE.search(k):
                return False
    return hf_moe_stacked_layout_default_from_transformers()
