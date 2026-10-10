"""CPU regression tests for optional Hugging Face RoPE scaling."""

import importlib.util
from pathlib import Path

import pytest
from transformers import PretrainedConfig


@pytest.fixture(scope="module")
def hf_config_utils():
    # Import this CPU-only helper without importing mbridge's GPU model registry.
    path = Path(__file__).resolve().parents[1] / "mbridge/utils/hf_config.py"
    spec = importlib.util.spec_from_file_location("hf_config_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("nested", [False, True])
def test_none_scaling_preserves_unscaled_defaults(hf_config_utils, nested):
    text_config = PretrainedConfig()
    text_config.rope_scaling = None
    config = PretrainedConfig(text_config=text_config) if nested else text_config

    scaling = hf_config_utils.get_hf_rope_scaling(config)
    defaults = {"type": "rope", "factor": 1, "mscale": 1.0}
    defaults.update(scaling)

    assert scaling == {}
    assert defaults == {"type": "rope", "factor": 1, "mscale": 1.0}
    assert text_config.rope_scaling is None


@pytest.mark.parametrize("nested", [False, True])
def test_populated_scaling_is_preserved(hf_config_utils, nested):
    scaling = {"type": "yarn", "factor": 16, "original_max_position_embeddings": 8192}
    text_config = PretrainedConfig()
    text_config.rope_scaling = scaling
    config = PretrainedConfig(text_config=text_config) if nested else text_config

    assert hf_config_utils.get_hf_rope_scaling(config) is scaling


def test_root_none_takes_precedence_over_nested_scaling(hf_config_utils):
    nested_scaling = {"type": "yarn", "factor": 16}
    text_config = PretrainedConfig()
    text_config.rope_scaling = nested_scaling
    config = PretrainedConfig(text_config=text_config)
    config.rope_scaling = None

    assert hf_config_utils.get_hf_rope_scaling(config) == {}
    assert config.text_config.rope_scaling is nested_scaling
