"""Regression test for split_deepstack_embs handling of None deepstack_visual_embeds.

When a microbatch contains only text samples (no images), the visual encoder
produces deepstack_visual_embeds=None while visual_pos_masks may still be a
non-None tensor. The early-return guard previously only checked visual_pos_masks,
so the function fell through to `for x in deepstack_visual_embeds` and crashed
with TypeError: 'NoneType' object is not iterable.

See ISEEKYAN/mbridge#47.
"""

import torch

from mbridge.models.qwen3_vl.utils import split_deepstack_embs


def test_returns_inputs_when_deepstack_is_none():
    """Pure-text microbatch: deepstack=None should NOT crash."""
    visual_pos_masks = torch.zeros(1, 8, dtype=torch.bool)

    out_masks, out_embeds = split_deepstack_embs(
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=None,
        tp_size=2,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        sequence_parallel=True,
    )

    assert out_embeds is None
    assert torch.equal(out_masks, visual_pos_masks)
