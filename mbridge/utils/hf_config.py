# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates

import warnings

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
    if hasattr(hf_config, "text_config") and hasattr(
        hf_config.text_config, "rope_theta"
    ):
        return hf_config.text_config.rope_theta

    # For transformers >= 5.0.0, check rope_parameters dict (optionally nested) for rope_theta
    rp = None
    if hasattr(hf_config, "rope_parameters"):
        rp = hf_config.rope_parameters
    elif hasattr(hf_config, "text_config") and hasattr(
        hf_config.text_config, "rope_parameters"
    ):
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


def get_hf_rope_scaling(hf_config: PretrainedConfig) -> dict:
    # For transformers <= 4.57.6
    if hasattr(hf_config, "rope_scaling"):
        return hf_config.rope_scaling
    if hasattr(hf_config, "text_config") and hasattr(
        hf_config.text_config, "rope_scaling"
    ):
        return hf_config.text_config.rope_scaling
    warnings.warn(
        f"rope_scaling not found in {type(hf_config).__name__}, keys: {hf_config.keys()}"
    )
    return {}


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
