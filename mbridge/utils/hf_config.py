# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates

import importlib.metadata
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


def hf_moe_checkpoint_uses_stacked_expert_weights() -> bool:
    """True when Hugging Face MoE checkpoints use fused expert tensors (transformers >= 5.0.0).

    In that layout, ``gate_up_proj`` has shape ``(num_experts, 2 * ffn_dim, hidden)`` and
    ``down_proj`` has shape ``(num_experts, hidden, ffn_dim)`` instead of per-expert modules.
    """
    return is_transformers_version_in_range("5.0.0", None)


def hf_moe_use_stacked_weights_for_checkpoint(
    index_keys: Optional[set[str]],
) -> bool:
    """Infer MoE layout from **on-disk** safetensors keys (used during :meth:`Bridge.load_weights`).

    **Legacy (experts.{i} API):** per-expert tensors such as
    ``…mlp.experts.{i}.gate_proj.weight``, ``up_proj.weight``, ``down_proj.weight``.

    **Current / transformers ≥5 style:** fused tensors
    ``…mlp.experts.gate_up_proj`` (and optional ``.weight``) and ``…mlp.experts.down_proj``.

    If ``index_keys`` is non-empty, presence of either pattern on layer 0 decides; otherwise
    fall back to :func:`hf_moe_checkpoint_uses_stacked_expert_weights`.
    """
    if index_keys:
        stacked = (
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.gate_up_proj.weight",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.0.mlp.experts.gate_up_proj.weight",
            "thinker.model.layers.0.mlp.experts.gate_up_proj",
            "thinker.model.layers.0.mlp.experts.gate_up_proj.weight",
        )
        if any(k in index_keys for k in stacked):
            return True
        per_expert = (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
            "thinker.model.layers.0.mlp.experts.0.gate_proj.weight",
        )
        if any(k in index_keys for k in per_expert):
            return False
    return hf_moe_checkpoint_uses_stacked_expert_weights()


def hf_moe_export_should_use_stacked_state_dict(extra_args: Optional[dict]) -> bool:
    """Target HuggingFace **state_dict** layout for Megatron → HF export (not load).

    - If ``extra_args`` contains ``hf_moe_export_stacked_expert_weights`` (bool), that value
      is used (force stacked fused tensors vs per-expert ``experts.{i}.*`` keys).
    - Otherwise: stacked when :func:`hf_moe_checkpoint_uses_stacked_expert_weights` is True
      (typically transformers ≥ 5), else legacy per-expert names.
    """
    if extra_args is not None and "hf_moe_export_stacked_expert_weights" in extra_args:
        return bool(extra_args["hf_moe_export_stacked_expert_weights"])
    return hf_moe_checkpoint_uses_stacked_expert_weights()