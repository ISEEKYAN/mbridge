from types import SimpleNamespace

import pytest
from transformers import LlamaConfig, MistralConfig

from mbridge import AutoBridge
from mbridge.models import LLaMABridge, MistralBridge


def _gptmodel_args(bridge_cls, hf_config):
    return bridge_cls._get_gptmodel_args(SimpleNamespace(hf_config=hf_config))


def test_unscaled_rope_has_no_interpolation_factor():
    # transformers>=5 reports {"rope_theta": ..., "rope_type": "default"} here.
    args = _gptmodel_args(LLaMABridge, LlamaConfig(rope_theta=1e6))
    assert args["rotary_base"] == 1e6
    assert "seq_len_interpolation_factor" not in args


def test_linear_rope_scaling_keeps_factor():
    hf_config = LlamaConfig(rope_scaling={"rope_type": "linear", "factor": 4.0})
    args = _gptmodel_args(LLaMABridge, hf_config)
    assert args["seq_len_interpolation_factor"] == 4.0


def test_mistral_is_registered():
    assert "mistral" in AutoBridge.list_supported_models()
    hf_config = MistralConfig(rope_theta=1e6, sliding_window=None)
    args = _gptmodel_args(MistralBridge, hf_config)
    assert args["rotary_base"] == 1e6
    assert "seq_len_interpolation_factor" not in args


def test_mistral_rejects_sliding_window():
    hf_config = MistralConfig(sliding_window=4096, max_position_embeddings=32768)
    with pytest.raises(NotImplementedError, match="sliding_window"):
        MistralBridge._build_config(SimpleNamespace(hf_config=hf_config))
