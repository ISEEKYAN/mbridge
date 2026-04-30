import logging
from typing import Optional

import torch
from megatron.core import InferenceParams, mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers import AutoConfig

from mbridge.core.util import (
    AllGatherVisionEmbeddings,
    collapse_thw,
    get_vision_cp_data,
    preprocess_packed_seqs,
    qwen3vl_cp_split,
    split_data_cp_rank,
)
from mbridge.models.qwen3_5.attention import Qwen3_5VLSelfAttention, SelfAttention
from mbridge.models.qwen3_5.transformer_config import Qwen3_5VLTransformerConfig
from mbridge.models.qwen3_5.utils import reorganize_inputs
from mbridge.models.qwen3_5.rope_utils import (
    Qwen3VLMultimodalRotaryEmbedding,
    get_rope_index,
)


class _SPScatterEmbeddingWrapper:
    """Thin callable that wraps a LanguageModelEmbedding and optionally scatters its output.

    Stored in the instance ``__dict__`` of ``Qwen3_5GPTModel`` (via
    ``object.__setattr__``), shadowing the real ``nn.Module`` registered in
    ``self._modules['embedding']``.  This means:
    - ``self.embedding(...)`` → wrapper (used by MTP, scatter-aware).
    - ``self._modules['embedding'](...)`` → real module, full unscattered output
      (used by ``Qwen3_5VLModel.forward`` before vision-token insertion).
    - Parameter names stay as ``language_model.embedding.*`` (weight loading intact).

    ``do_scatter`` is immutable after construction.  If the training-job data
    format is not known at ``__init__`` time, create with ``do_scatter=True``
    (the default) and replace the wrapper once on the first forward call via
    ``Qwen3_5GPTModel.init_mtp_embedding_scatter``.
    """

    def __init__(self, real_embedding, do_scatter: bool):
        self._real_embedding = real_embedding
        self._do_scatter = do_scatter

    def __call__(self, **kwargs):
        out = self._real_embedding(**kwargs)
        if self._do_scatter:
            out = tensor_parallel.scatter_to_sequence_parallel_region(out)
            out = out.contiguous()
        return out

    # Proxy attribute access so code that does ``self.embedding.weight`` still works.
    def __getattr__(self, name):
        return getattr(self._real_embedding, name)


class Qwen3_5GPTModel(GPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=kwargs["rotary_percent"],
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=kwargs.get(
                "seq_len_interpolation_factor", None
            ),
            rotary_base=kwargs.get("rotary_base", 10000),
        )
        self.mrope_section = self.config.mrope_section
        assert (
            self.mrope_section is not None
        ), "mrope require mrope_section setting, but we got None from TransformerConfig"

        # ``self.embedding`` (LanguageModelEmbedding) only exists on PP stages where
        # pre_process or mtp_process is True; it is None on all other stages.
        # Guard to avoid wrapping None.
        if self._modules.get("embedding") is not None:
            # VL models initialise their language model with
            # ``scatter_embedding_sequence_parallel=False`` so that Qwen3_5VLModel.forward
            # can obtain the full [s, b, h] embedding before inserting vision tokens and
            # manually packing/scattering it.  The side-effect is that MTP also receives
            # a full (unscattered) embedding that mismatches the SP-sharded hidden_states
            # produced by the decoder, causing a size-mismatch in _concat_embeddings.
            #
            # Fix: shadow ``self.embedding`` in the instance __dict__ with a wrapper
            # that applies SP-scatter on output.  The real nn.Module stays registered
            # in ``self._modules['embedding']`` so that parameter names (used by weight
            # loading) remain ``language_model.embedding.*`` unchanged.
            # The scatter behaviour is determined by the data format (THD vs BSHD) which
            # is not known at __init__ time, so we start with do_scatter=True (THD default)
            # and let Qwen3_5VLModel.forward replace the wrapper once on the first call.
            object.__setattr__(
                self,
                "embedding",
                _SPScatterEmbeddingWrapper(self._modules["embedding"], do_scatter=self.config.sequence_parallel),
            )

    def init_mtp_embedding_scatter(self, do_scatter: bool) -> None:
        """Replace the MTP embedding wrapper with the correct scatter setting.

        Called once by ``Qwen3_5VLModel.forward`` on the first forward pass, after
        the data format (THD vs BSHD) is known.  Subsequent calls are no-ops.
        """
        if "embedding" not in self.__dict__:
            return
        wrapper = object.__getattribute__(self, "embedding")  # bypasses __getattr__
        if not isinstance(wrapper, _SPScatterEmbeddingWrapper):
            return
        if wrapper._do_scatter == do_scatter:
            return  # already correct, nothing to do
        object.__setattr__(
            self,
            "embedding",
            _SPScatterEmbeddingWrapper(wrapper._real_embedding, do_scatter=do_scatter),
        )


class Qwen3_5VLModel(MegatronModule):
    """Qwen3_5VLModel model"""

    def __init__(
        self,
        language_transformer_config: Qwen3_5VLTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        hf_config: AutoConfig,
        hf_vision_cls: type,
        language_mtp_block_spec: Optional[ModuleSpec] = None,
        parallel_output: bool = True,
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        position_embedding_type: str = "mrope",
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        rope_scaling: bool = False,
    ) -> None:
        super().__init__(config=language_transformer_config)

        for _, layer_spec in enumerate(language_transformer_layer_spec.layer_specs):
            # only replace SelfAttention
            if issubclass(layer_spec.submodules.self_attention.module, SelfAttention):
                layer_spec.submodules.self_attention.module = Qwen3_5VLSelfAttention

        if language_mtp_block_spec is not None:
            for _, layer_spec in enumerate(language_mtp_block_spec.layer_specs):
                # 'mtp_model_layer' is the new name in megatron (renamed from 'transformer_layer')
                mtp_inner = getattr(layer_spec.submodules, 'mtp_model_layer', None) or getattr(layer_spec.submodules, 'transformer_layer', None)
                if mtp_inner is not None and issubclass(mtp_inner.submodules.self_attention.module, SelfAttention):
                    mtp_inner.submodules.self_attention.module = Qwen3_5VLSelfAttention


        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.hf_config = hf_config
        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        self.square_merge_size = self.spatial_merge_size**2

        if self.pre_process:
            self.vision_model = hf_vision_cls._from_config(hf_config.vision_config)
            self._hook_fp32_rotary_emb(self.vision_model)
            self._hook_vision_params_avg_grad_across_tp(self.vision_model)

        self.language_model = Qwen3_5GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=language_rotary_percent,
            rotary_base=language_rotary_base,
            rope_scaling=rope_scaling,
            mtp_block_spec=language_mtp_block_spec,
            scatter_embedding_sequence_parallel=False,
        )

    @staticmethod
    def _hook_fp32_rotary_emb(module: torch.nn.Module):
        """Force all RotaryEmbedding inv_freq buffers to stay at original float32 precision.
        """
        for submodule in module.modules():
            if hasattr(submodule, "inv_freq") and submodule.inv_freq is not None:
                # Save the original float32 inv_freq (this runs BEFORE Float16Module)
                submodule._inv_freq_fp32_original = (
                    submodule.inv_freq.detach().clone().float()
                )

                def _hook(mod, args):
                    if hasattr(mod, "_inv_freq_fp32_original"):
                        # Restore inv_freq from the saved fp32_copied
                        mod.inv_freq = mod._inv_freq_fp32_original.to(
                            device=mod.inv_freq.device
                        )

                submodule.register_forward_pre_hook(_hook)

    def _hook_vision_params_avg_grad_across_tp(self, module: torch.nn.Module):
        """Mark all vision model parameters with average_gradients_across_tp_domain=True.

        Since the vision model receives full (non-SP-split) input on every TP rank,
        each TP rank computes the full gradient independently. Using AVG all-reduce
        across TP ranks keeps replicated weights synchronized without scaling the
        gradient by tp_size (which SUM would incorrectly do).
        """
        if module is None:
            return module
        for param in module.parameters(recurse=True):
            setattr(param, "average_gradients_across_tp_domain", True)

    @property
    def share_embeddings_and_output_weights(self):
        return self.language_model.share_embeddings_and_output_weights

    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"

        self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model is not None:
            modules.append(self.vision_model.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        if freeze_vision_model and not freeze_vision_projection:
            if self.vision_model is not None:
                for param in self.vision_model.merger.parameters():
                    param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        # can set at dataset
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        cp_img_num: list[int] = None,
        images_padded: list[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert inference_context is None, "not support inference yet"

        vision_grid_thw = None
        vision_data = None
        vision_mask = None
        # TODO: this approach may need rethinking
        cp_size = mpu.get_context_parallel_world_size()

        # Track packed input_ids (THD format) for MTP use when remove_padding is enabled
        input_ids_packed = None

        if self.pre_process:
            # can reorganize_inputs at dataset
            vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                square_merge_size=self.square_merge_size,
            )

            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                if cp_size > 1:
                    if cp_img_num is None:
                        assert images_padded is None
                        vision_data, vision_grid_thw, cp_img_num, images_padded = (
                            qwen3vl_cp_split(
                                cp_size,
                                vision_data,
                                vision_grid_thw,
                            )
                        )
                    vision_data, vision_grid_thw, seqlen_on_cp_ranks = (
                        get_vision_cp_data(
                            vision_data,
                            vision_grid_thw,
                            self.square_merge_size,
                            cp_img_num,
                            images_padded,
                        )
                    )
                    vision_grid_thw = collapse_thw(vision_grid_thw)
                if vision_data.shape[0] > 0:
                    vision_embeds = self.vision_model(
                        hidden_states=vision_data,
                        grid_thw=vision_grid_thw,
                    ).pooler_output
                    # Encodes images into continuous embeddings that can be forwarded to the language model.
                    split_sizes = (
                        vision_grid_thw.prod(-1) // self.spatial_merge_size**2
                    ).tolist()
                    vision_embeds = torch.split(vision_embeds, split_sizes)
                    vision_embeds = torch.cat(vision_embeds, dim=0)
                else:
                    vision_embeds = torch.zeros(
                        (0, self.language_model.config.hidden_size),
                        device=vision_data.device,
                        dtype=torch.bfloat16,
                    )
                if cp_size > 1:
                    vision_embeds = AllGatherVisionEmbeddings.apply(
                        vision_embeds,
                        seqlen_on_cp_ranks,
                    )

            combined_embeddings = self.language_model._modules["embedding"](
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if vision_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[vision_mask] = vision_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if (
                combined_embeddings is not None
                and cp_size > 1
                and packed_seq_params is None
            ):
                combined_embeddings = split_data_cp_rank(
                    combined_embeddings, cp_size, 0
                )
            if packed_seq_params is not None:
                input_ids_thd, _ = preprocess_packed_seqs(
                    input_ids, attention_mask, pre_process=True
                )
                # Save packed input_ids for MTP use outside this block
                input_ids_packed = input_ids_thd
                vision_mask_thd = (input_ids_thd == self.image_token_id) | (
                    input_ids_thd == self.video_token_id
                )

                vision_mask = vision_mask_thd
                # combined_embeddings is [s, b, h] (SBH); convert to THD format [total_tokens, 1, h]:
                # 1. transpose to [b, s, h], then preprocess_packed_seqs → [1, total_tokens, h]
                #    (preprocess_packed_seqs always unsqueeze(0) before return)
                # 2. squeeze(0) → [total_tokens, h]
                # 3. unsqueeze(1) → [total_tokens, 1, h] (THD format), so scatter_to_sequence_parallel
                #    acts on the token dim (dim 0) and produces [total_tokens//tp, 1, h]
                combined_embeddings_thd = (
                    preprocess_packed_seqs(
                        combined_embeddings.transpose(0, 1).contiguous(),
                        attention_mask,
                        pre_process=True,
                    )[0]
                    .squeeze(0)
                    .unsqueeze(1)
                    .contiguous()
                )
                combined_embeddings = combined_embeddings_thd

            if self.config.sequence_parallel:
                combined_embeddings = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        combined_embeddings
                    )
                )
                combined_embeddings = combined_embeddings.contiguous()

        else:
            combined_embeddings = None

        # Save the original attention_mask before it may be set to None in the THD branch
        # below. Non-pre_process stages need it to generate packed input_ids for MTP.
        attention_mask_orig = attention_mask

        if position_ids is None:
            # BSHD
            position_ids, _ = get_rope_index(
                self.config.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )  #  [3*b*s]
            if packed_seq_params is not None:
                # convert position_ids to THD format
                position_ids = (
                    preprocess_packed_seqs(
                        position_ids.permute(1, 2, 0), attention_mask, pre_process=True
                    )[0]
                    .permute(2, 0, 1)
                    .contiguous()
                )
                attention_mask = None
                self.language_model.rotary_pos_emb.is_thd_format = True

        # Prepare input_ids for language_model (needed by MTP).
        # When packed_seq_params is not None (remove_padding enabled), use the already-packed
        # input_ids whose total token count is aligned to tp_size (guaranteed by
        # preprocess_packed_seqs). Scattering the original padded input_ids would fail because
        # padded seq_len is not guaranteed to be divisible by tp_size.
        # When packed_seq_params is None, scatter the padded input_ids along the sequence dim
        # so each TP rank holds seq_len // tp_size tokens, consistent with SP hidden_states.
        
        if not self.language_model.config.mtp_num_layers:
            # MTP is not enabled; skip all input_ids preparation for MTP.
            sp_input_ids = None
        else:
            if packed_seq_params is not None:
                # THD mode: use packed input_ids whose total token count is aligned to tp_size.
                if input_ids_packed is not None:
                    sp_input_ids = input_ids_packed
                elif attention_mask_orig is not None:
                    # pre_process=False (non-first PP stage): input_ids_packed was not generated
                    # in the pre_process block above, but MTP still needs packed input_ids whose
                    # token count matches the THD hidden_states.  Re-pack here using the original
                    # attention_mask (before it was cleared to None for the decoder).
                    sp_input_ids, _ = preprocess_packed_seqs(
                        input_ids, attention_mask_orig, pre_process=True
                    )
                else:
                    sp_input_ids = input_ids

                # THD mode: embedding wrapper must scatter (do_scatter=True).
                self.language_model.init_mtp_embedding_scatter(do_scatter=True)
            else:
                sp_input_ids = input_ids

                if input_ids is not None and combined_embeddings is not None:
                    # BSHD mode: input_ids may arrive already SP-scattered (shape [B, S/TP])
                    # or still full-length (shape [B, S]).  combined_embeddings has already been
                    # CP-split and TP-scattered by this point, so its dim-0 equals S/(TP*CP*2).
                    # We need sp_input_ids to be [B, S/(TP*CP*2)] so that MTP embedding output
                    # [S/(TP*CP*2), B, H] matches hidden_states [S/(TP*CP*2), B, H].
                    # Step 1: TP scatter along seq dim: [B, S] -> [B, S/TP]
                    # Step 2: CP split along seq dim:   [B, S/TP] -> [B, S/(TP*CP*2)]
                    if input_ids.shape[1] != combined_embeddings.shape[0]:
                        sp_input_ids = input_ids.permute(1, 0).contiguous()
                        sp_input_ids = tensor_parallel.scatter_to_sequence_parallel_region(sp_input_ids)
                        if cp_size > 1:
                            # CP split: [S/TP, B] -> [S/(TP*CP*2), B]
                            sp_input_ids = split_data_cp_rank(sp_input_ids, cp_size, seq_dim=0)
                        sp_input_ids = sp_input_ids.permute(1, 0).contiguous()

                    # sp_input_ids is now [B, S/(TP*CP*2)]: embedding wrapper must NOT scatter again.
                    self.language_model.init_mtp_embedding_scatter(do_scatter=False)
                    wrapper_after = object.__getattribute__(self.language_model, "embedding") if "embedding" in self.language_model.__dict__ else None


        return self.language_model(
            input_ids=sp_input_ids,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,  # currently always None
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,  # currently always None
            **(extra_block_kwargs or {}),
            **kwargs,
        )

    def verify_vision_weights_consistency(self, step: int = -1):
        """
        Verify that vision model weights are consistent across all ranks.
        Should be called after optimizer.step() and param all-gather.

        Checks:
        1. Across TP ranks: vision weights should be identical (replicated, not sharded).
        2. Across DP ranks: vision weights should be identical (same optimizer update).
        """
        if self.vision_model is None:
            return

        rank = torch.distributed.get_rank()
        tp_group = mpu.get_tensor_model_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp_group = mpu.get_data_parallel_group()
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()

        logger = logging.getLogger(__name__)
        all_consistent = True

        for name, param in self.vision_model.named_parameters():
            if not param.requires_grad:
                continue

            # --- Check 1: TP consistency ---
            if tp_size > 1:
                # Gather weights from all TP ranks
                gathered = [torch.zeros_like(param.data) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered, param.data, group=tp_group)
                for i in range(1, tp_size):
                    if not torch.equal(gathered[0], gathered[i]):
                        max_diff = (gathered[0] - gathered[i]).abs().max().item()
                        if torch.distributed.get_rank() == 0:
                            logger.warning(
                                f"[Step {step}] TP MISMATCH: vision_model.{name} "
                                f"tp_rank 0 vs tp_rank {i}, max_diff={max_diff:.6e}"
                                f" {gathered[0].sum()} {gathered[i].sum()}"
                            )
                        all_consistent = False

            # --- Check 2: DP consistency ---
            if dp_size > 1:
                gathered = [torch.zeros_like(param.data) for _ in range(dp_size)]
                torch.distributed.all_gather(gathered, param.data, group=dp_group)
                for i in range(1, dp_size):
                    if not torch.equal(gathered[0], gathered[i]):
                        max_diff = (gathered[0] - gathered[i]).abs().max().item()
                        if torch.distributed.get_rank() == 0:
                            logger.warning(
                                f"[Step {step}] DP MISMATCH: vision_model.{name} "
                                f"dp_rank 0 vs dp_rank {i}, max_diff={max_diff:.6e}"
                                f" {gathered[0].sum()} {gathered[i].sum()}"
                            )
                        all_consistent = False

        if all_consistent and rank == 0:
            logger.info(
                f"[Step {step}] matching. Vision model weights consistent across "
                f"TP({tp_size}) and DP({dp_size}) ranks."
            )
