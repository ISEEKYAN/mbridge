from typing import Optional, Union, Tuple
import torch
from torch import Tensor

from megatron.core.transformer.attention import *

from mbridge.models.qwen3_vl.rope_utils import apply_rotary_pos_emb_absolute


class Qwen3p5VLSelfAttention(SelfAttention):
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                "2.7.3"
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        is_inference_mode = inference_context is not None and not self.training
        # is_using_flash_decode - True is we are using the static inference engine with flash decode
        is_using_flash_decode = is_inference_mode and self.config.flash_decode
        # is_using_flashinfer_rope - True if we are using the dynamic inference engine
        # with flashinfer fused rope
        is_using_flashinfer_rope = is_inference_mode and (
            not inference_context.is_static_batching()
            and inference_context.use_flashinfer_fused_rope
        )
        if is_using_flash_decode or is_using_flashinfer_rope:
            # flash decode and flash-infer fused rope use rotary_pos_cos and rotary_pos_sin
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix="qkv")
        split_qkv = (self.attention_type == "cross") or not all(
            [
                not self.config.test_mode,
                self.config.fused_single_qkv_rope,
                inference_context is None,
                packed_seq_params is None,
                (
                    rotary_pos_emb is not None
                    and rotary_pos_emb[0] is not None
                    and rotary_pos_emb[1] is not None
                ),
                not self.config.flash_decode,
                HAVE_FUSED_QKV_ROPE,
                self.q_layernorm is None or isinstance(self.q_layernorm, IdentityOp),
                self.k_layernorm is None or isinstance(self.k_layernorm, IdentityOp),
            ]
        )
        output_gate = self.config.attention_output_gate
        # Check if fused_single_qkv_rope is requested but either unavailable or not
        # supported for the current use case.
        if self.attention_type != "cross":
            assert not (
                self.config.fused_single_qkv_rope and split_qkv
            ), "fused_single_qkv_rope requested but not available/supported for the config."
        if output_gate:
            assert split_qkv, "output_gate is not supported for unsplit mixed_qkv tensor."

        with off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear") as hidden_states:
            qkv_output = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                split_qkv=split_qkv,
                output_gate=self.config.attention_output_gate,
            )
        if self.offload_qkv_linear:
            # `qkv_output` may be a tuple; commit supports tuple/list and will keep structure.
            qkv_output = off_interface.group_commit(
                qkv_output, name="qkv_linear", forced_released_tensors=[]
            )
        attn_mask_type = self.attn_mask_type
        block_table = None
        gate = None
        if split_qkv:
            if self.config.attention_output_gate:
                query, key, value, gate = qkv_output
            else:
                query, key, value = qkv_output
            mixed_qkv = qkv_split_arg_list = None
        else:
            assert (
                not self.config.attention_output_gate
            ), "attention_output_gate is not supported for unsplit mixed_qkv tensor."
            mixed_qkv, qkv_split_arg_list = qkv_output
        nvtx_range_pop(suffix="qkv")

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (
            inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
        )

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix="adjust_key_value")
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if (
            in_decode_mode
            and self.config.cuda_graph_impl == "local"
            and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
            and inference_context.is_static_batching()
        ):
            raise ValueError(f"CUDA graphs must use flash decode with static batching!")

        if split_qkv:
            query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
                self._adjust_key_value_for_inference(
                    inference_context,
                    query,
                    key,
                    value,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    rotary_pos_cos_sin,
                    sequence_len_offset,
                )
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and (
            not self.config.flash_decode or inference_context is None
        ):
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if split_qkv:
                if q_pos_emb is not None:
                    # TODO VIJAY: simplify
                    if inference_context is None or inference_context.is_static_batching():
                        query = apply_rotary_pos_emb_absolute(
                            query,
                            q_pos_emb,
                            config=self.config,
                            cu_seqlens=cu_seqlens_q,
                            mscale=_yarn_get_concentration_factor_from_config(self.config),
                            cp_group=self.pg_collection.cp,
                        )
                    else:
                        query = inference_context.apply_rotary_emb_query(
                            query, q_pos_emb, self.config, cu_seqlens_q, self.pg_collection.cp
                        )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb_absolute(
                        key,
                        k_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
            else:
                raise ValueError("fused_qkv_rotary_pos_emb is not supported for unsplit mixed_qkv tensor.")

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix="core_attention")
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                with off_interface(
                    self.offload_core_attention and self.training, query, "core_attn"
                ) as query:
                    core_attn_out = apply_module(self.core_attention)(
                        query,
                        key,
                        value,
                        attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                    inference_context.is_decode_only(),
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

                # Clear the outputs for padding tokens when using quantization scales
                # to avoid corrupting amax calculations
                if is_using_quantization_scales(self.config):
                    core_attn_out[inference_context.padding_slice] = 0.0

            if self.offload_core_attention and self.training:
                core_attn_out = off_interface.group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        # Output gate
        if gate is not None:
            nvtx_range_push(suffix="output_gate")
            core_attn_out = self._apply_output_gate(core_attn_out, gate)
            nvtx_range_pop(suffix="output_gate")

        # =================
        # Output. [sq, b, h]
        # =================
        nvtx_range_push(suffix="linear_proj")
        with off_interface(self.offload_attn_proj, core_attn_out, "attn_proj") as core_attn_out:
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output = off_interface.group_commit(
                output, name="attn_proj", forced_released_tensors=[core_attn_out]
            )
        nvtx_range_pop(suffix="linear_proj")

        return output, bias
