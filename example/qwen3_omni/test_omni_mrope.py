import torch
from torch import nn

try:
    from transformers import Qwen3OmniMoeThinkerConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )
except:
    print(f"your install the tranformers>=4.57.0 or install from source")
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

from example.qwen3_omni.load_model_and_forward import get_sample_for_forward
from mbridge.models.qwen3_vl.rope_utils import Qwen3VLMultimodalRotaryEmbedding
from mbridge.models.qwen3_omni_moe.rope_utils import get_rope_index


def init_q_k(hf_config, token_ids):
    head_num = hf_config.text_config.num_attention_heads
    dim = hf_config.text_config.head_dim
    bsz = token_ids.shape[0]
    q_len = token_ids.shape[1]
    q = torch.randn(bsz, head_num, q_len, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(bsz, head_num, q_len, dim, dtype=torch.bfloat16, device="cuda")
    return q, k


# PYTHONPATH=${PWD}:/root/Megatron-LM:$PYTHONPATH python example/qwen3_omni/test_omni_mrope.py
if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.set_default_device("cuda:0")
    hf_model_path = "/data/model/Qwen3-Omni-30B-A3B-Instruct"

    inputs = get_sample_for_forward(hf_model_path)
    hf_config = Qwen3OmniMoeThinkerConfig.from_pretrained(hf_model_path)

    audio_feature_lengths = torch.sum(inputs["feature_attention_mask"], dim=1)
    position_ids, _ = get_rope_index(
        hf_config.vision_config.spatial_merge_size,
        hf_config.image_token_id,
        hf_config.video_token_id,
        hf_config.audio_token_id,
        hf_config.vision_start_token_id,
        hf_config.audio_start_token_id,
        hf_config.position_id_per_seconds,
        inputs["input_ids"],
        inputs["image_grid_thw"],
        None,
        inputs["attention_mask"],
        False,
        audio_feature_lengths,
        None,
    )
    hf = Qwen3OmniMoeThinkerTextRotaryEmbedding(hf_config.text_config)

    mlm = Qwen3VLMultimodalRotaryEmbedding(
        kv_channels=hf_config.text_config.head_dim,
        rotary_percent=1.0,
        rotary_interleaved=False,
        seq_len_interpolation_factor=None,
        rotary_base=hf_config.text_config.rope_theta,
    )
    x = torch.randn([128, 256], dtype=torch.bfloat16, device="cuda:0")
    hf_cos, hf_sin = hf(x, position_ids)

    mrope_section = [24, 20, 20]
    emb = mlm(position_ids, mrope_section)

    cos_ = (torch.cos(emb)).to(x.dtype)
    sin_ = (torch.sin(emb)).to(x.dtype)
    assert torch.equal(hf_cos.squeeze(), cos_.squeeze())
    assert torch.equal(hf_sin.squeeze(), sin_.squeeze())

    q, k = init_q_k(hf_config, inputs["input_ids"])
    hf_q, hf_k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    mlm_q = _apply_rotary_pos_emb_bshd(q.permute(2, 0, 1, 3), emb)
    mlm_k = _apply_rotary_pos_emb_bshd(k.permute(2, 0, 1, 3), emb)
    mlm_q = mlm_q.permute(1, 2, 0, 3)
    mlm_k = mlm_k.permute(1, 2, 0, 3)

    assert torch.equal(hf_k, mlm_k)
    assert torch.equal(hf_q, mlm_q)
    print("test_mrope passed")
