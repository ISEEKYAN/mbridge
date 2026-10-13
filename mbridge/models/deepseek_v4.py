# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek-V4 bridge for mbridge.

Translates the DeepSeek-V4 (Flash and Pro) HuggingFace config into a
Megatron-Core ``MLATransformerConfig`` that wires the experimental
``dsv4_hybrid`` self-attention variant (CSA + HCA + DSA indexer),
per-layer hyper-connections (mHC), hash-routed MoE with ClampedSwiGLU,
and optional Multi-Token Prediction (MTP) layers.

Scope and limitations
---------------------
- Targets Megatron-LM commits that include the DSv4 hybrid attention
  variant, CSA/HCA, DSA indexer, mHC fused kernels, hash routing, and
  ClampedSwiGLU. These currently live on the ``dev`` branch.
- TP must be 1: ``DSv4HybridSelfAttention`` asserts
  ``tensor_model_parallel_size == 1`` upstream.
- DSv4 hybrid attention does not accept ``inference_context``; the
  bridge supports training paths only. vLLM / SGLang serve inference
  through a separate kernel path.
- CSA/DSA do not yet accept ``packed_seq_params``; THD and CP are not
  enabled in this bridge.
- Full FP8 / MXFP4 dequantisation for DSv4-Flash post-trained
  checkpoints is **not** implemented here. Use
  ``NVIDIA-NeMo/Megatron-Bridge`` for that path. This bridge currently
  covers config translation, model build from a HuggingFace config
  object, and a partial weight-name mapping for Megatron-trained
  checkpoints; advanced parameters (mHC alpha scalars, DSA indexer
  weights, compressor weights, hash-routing ``tid2eid`` buffer,
  attention sink) raise ``NotImplementedError`` so callers know the
  boundary.
"""

import dataclasses
import inspect
from typing import Callable, Optional

import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import MLATransformerConfig

from ..core import LLMBridge, register_model
from ..utils.hf_config import get_hf_rope_theta

# --- DSv4 layer-type / compress-ratio bookkeeping -------------------------

_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _dsv4_num_hash_layers(hf_config) -> int:
    """Return the number of contiguous hash-MoE layers at the start of the model.

    DeepSeek-V4 places its hash-routing layers as a contiguous prefix in
    ``mlp_layer_types``. Newer HF configs may instead expose ``num_hash_layers``
    directly; both forms are supported.
    """
    num_hash_layers = getattr(hf_config, "num_hash_layers", None)
    if num_hash_layers is not None:
        return int(num_hash_layers)

    mlp_layer_types = getattr(hf_config, "mlp_layer_types", None)
    if mlp_layer_types is None:
        return 0

    n_hash = 0
    for layer_type in mlp_layer_types:
        if layer_type != "hash_moe":
            break
        n_hash += 1
    if any(layer_type == "hash_moe" for layer_type in mlp_layer_types[n_hash:]):
        raise ValueError("DeepSeek-V4 hash MoE layers must be a contiguous prefix.")
    return n_hash


def _dsv4_compress_ratios(hf_config) -> list[int]:
    """Translate the DSv4 layer schedule to MCore ``csa_compress_ratios``.

    The HuggingFace config exposes the per-layer attention schedule either as
    a flat ``compress_ratios`` list (legacy form) or as ``layer_types`` plus
    ``compress_rates`` (newer Transformers native form). MCore consumes a
    flat list whose length equals ``num_hidden_layers + num_nextn_predict_layers``.
    """
    num_hidden_layers = int(hf_config.num_hidden_layers)
    num_mtp_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
    expected_len = num_hidden_layers + num_mtp_layers

    compress_ratios = getattr(hf_config, "compress_ratios", None)
    if compress_ratios is not None:
        ratios = [int(ratio) for ratio in compress_ratios]
    else:
        layer_types = getattr(hf_config, "layer_types", None)
        compress_rates = getattr(hf_config, "compress_rates", None)
        if layer_types is None or compress_rates is None:
            raise ValueError(
                "HF config missing 'compress_ratios' and native "
                "'layer_types' / 'compress_rates'. DeepSeek-V4 requires per-layer "
                "compression ratios."
            )
        ratios = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                ratios.append(0)
            elif layer_type in compress_rates:
                ratios.append(int(compress_rates[layer_type]))
            elif layer_type in _DSV4_LAYER_TYPE_TO_COMPRESS_RATIO:
                ratios.append(_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO[layer_type])
            else:
                raise ValueError(
                    f"Unsupported DeepSeek-V4 attention layer type: {layer_type!r}"
                )

    # If config omitted MTP entries, pad with zeros (sliding-attention).
    if len(ratios) == num_hidden_layers and num_mtp_layers:
        ratios.extend([0] * num_mtp_layers)
    if len(ratios) < expected_len:
        raise ValueError(
            f"DeepSeek-V4 compression ratios length ({len(ratios)}) is shorter than "
            f"num_hidden_layers + num_nextn_predict_layers ({expected_len})."
        )
    return ratios[:expected_len]


@register_model("deepseek_v4")
class DeepseekV4Bridge(LLMBridge):
    """Bridge implementation for DeepSeek-V4 (Flash / Pro) causal LM models.

    Notes
    -----
    DSv4 layers are heterogeneous: a contiguous prefix uses hash-MoE routing
    without a DSA indexer; the remainder uses ``compressed_sparse_attention``
    (ratio 4, with DSA indexer) or ``heavily_compressed_attention`` (ratio 128,
    compressor only). All layers share the MLA Q/KV projections and the
    factored output projection (``wo_a`` + ``wo_b``). Per-layer
    hyper-connections feed a learned output contraction (``hc_head_*``) on
    the TransformerBlock; the model_provider wires this through the
    experimental block spec builder.
    """

    TransformerConfigClass = MLATransformerConfig

    # DeepSeek-V4 native HF checkpoint naming differs from the
    # ``model.embed_tokens.*`` Transformers convention used by DSv3 — DSv4
    # checkpoints serialise as ``embed.weight``, ``head.weight``, etc.
    # This mapping reflects the native names. Full mHC / DSA / hash-routing
    # buffers are intentionally omitted; see class docstring.
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "embed.weight",
        "decoder.final_layernorm.weight": "norm.weight",
        "output_layer.weight": "head.weight",
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.layer_norm_weight": [
            "layers.{layer_number}.ffn_norm.weight",
        ],
        "pre_mlp_layernorm.weight": [
            "layers.{layer_number}.ffn_norm.weight",
        ],
        "mlp.router.weight": [
            "layers.{layer_number}.ffn.gate.weight",
        ],
        "mlp.router.expert_bias": [
            "layers.{layer_number}.ffn.gate.bias",
        ],
        # Routed experts (gate / up fused into linear_fc1, down in linear_fc2).
        "mlp.experts.linear_fc1.weight": [
            "layers.{layer_number}.ffn.experts.{expert_id}.w1.weight",
            "layers.{layer_number}.ffn.experts.{expert_id}.w3.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "layers.{layer_number}.ffn.experts.{expert_id}.w2.weight",
        ],
        # Shared expert.
        "mlp.shared_experts.linear_fc1.weight": [
            "layers.{layer_number}.ffn.shared_experts.w1.weight",
            "layers.{layer_number}.ffn.shared_experts.w3.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [
            "layers.{layer_number}.ffn.shared_experts.w2.weight",
        ],
    }

    # MLA with factored output projection (wo_a is a replicated nn.Parameter,
    # wo_b is a row-parallel linear).
    _ATTENTION_MAPPING = {
        "input_layernorm.weight": [
            "layers.{layer_number}.attn_norm.weight",
        ],
        "self_attention.linear_q_down_proj.weight": [
            "layers.{layer_number}.attn.wq_a.weight",
        ],
        "self_attention.linear_q_up_proj.layer_norm_weight": [
            "layers.{layer_number}.attn.q_norm.weight",
        ],
        "self_attention.q_layernorm.weight": [
            "layers.{layer_number}.attn.q_norm.weight",
        ],
        "self_attention.linear_q_up_proj.weight": [
            "layers.{layer_number}.attn.wq_b.weight",
        ],
        "self_attention.linear_kv_proj.weight": [
            "layers.{layer_number}.attn.wkv.weight",
        ],
        "self_attention.kv_layernorm.weight": [
            "layers.{layer_number}.attn.kv_norm.weight",
        ],
        "self_attention.linear_proj.weight": [
            "layers.{layer_number}.attn.wo_b.weight",
        ],
        # linear_o_group_proj is a plain nn.Parameter replicated across TP.
        "self_attention.linear_o_group_proj": [
            "layers.{layer_number}.attn.wo_a.weight",
        ],
    }

    # ------------------------------------------------------------------
    # Config translation
    # ------------------------------------------------------------------

    def _build_config(self):
        """Build the ``MLATransformerConfig`` for DeepSeek-V4 from the HF config."""
        hf_config = self.hf_config

        # MLA geometry. ``head_dim`` in the DSv4 HF config is the V head dim;
        # the q/k head dimensions are derived inside the DSv4 hybrid attention
        # module from ``v_head_dim`` and ``qk_pos_emb_head_dim``.
        v_head_dim = int(hf_config.head_dim)
        qk_pos_emb_head_dim = int(hf_config.qk_rope_head_dim)
        q_lora_rank = int(hf_config.q_lora_rank)
        o_groups = int(hf_config.o_groups)
        o_lora_rank = int(hf_config.o_lora_rank)

        # YaRN RoPE. DSv4 may carry a single rope_scaling dict (legacy) or
        # a split form with separate "main" and "compress" entries.
        rope_params = (
            getattr(hf_config, "rope_scaling", None)
            or getattr(hf_config, "rope_parameters", None)
            or {}
        )
        if "compress" in rope_params:
            main_rope = rope_params.get("main", {})
            compress_rope = rope_params["compress"]
        else:
            main_rope = rope_params
            compress_rope = rope_params
        rotary_base = float(main_rope.get("rope_theta", get_hf_rope_theta(hf_config)))
        csa_compress_rotary_base = float(
            compress_rope.get(
                "rope_theta",
                getattr(hf_config, "compress_rope_theta", rotary_base),
            )
        )
        rotary_scaling_factor = float(compress_rope.get("factor", 1.0))
        original_max_position_embeddings = int(
            compress_rope.get(
                "original_max_position_embeddings", hf_config.max_position_embeddings
            )
        )

        # Layer-type schedule for CSA / HCA.
        csa_compress_ratios = _dsv4_compress_ratios(hf_config)

        # MoE. All decoder layers are MoE in DSv4 (no dense prefix like V3).
        moe_layer_freq = [1] * int(hf_config.num_hidden_layers)
        num_hash_layers = _dsv4_num_hash_layers(hf_config)

        # MTP.
        mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or None
        mtp_args = {}
        if mtp_num_layers:
            mtp_args["mtp_num_layers"] = mtp_num_layers
            mtp_args["mtp_loss_scaling_factor"] = self.extra_args.get(
                "mtp_loss_scaling_factor", 0.1
            )

        base_config = {
            # --- DSv4 hybrid attention ---
            "experimental_attention_variant": "dsv4_hybrid",
            "multi_latent_attention": True,
            "qk_layernorm": True,
            "normalization": "RMSNorm",
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "add_bias_linear": False,
            "ffn_hidden_size": hf_config.intermediate_size,
            # MLA geometry (kv_channels is derived internally by DSv4 hybrid attn;
            # explicitly null it to avoid LLMBridge auto-mapping ``head_dim`` here).
            "kv_channels": None,
            "v_head_dim": v_head_dim,
            "qk_pos_emb_head_dim": qk_pos_emb_head_dim,
            "q_lora_rank": q_lora_rank,
            "o_groups": o_groups,
            "o_lora_rank": o_lora_rank,
            # --- RoPE / YaRN ---
            "rope_type": "yarn",
            "rotary_base": rotary_base,
            "csa_compress_rotary_base": csa_compress_rotary_base,
            "rotary_scaling_factor": rotary_scaling_factor,
            "original_max_position_embeddings": original_max_position_embeddings,
            "beta_fast": float(compress_rope.get("beta_fast", 32)),
            "beta_slow": float(compress_rope.get("beta_slow", 1)),
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "apply_rope_fusion": True,
            # --- CSA / DSA ---
            "csa_compress_ratios": csa_compress_ratios,
            "csa_window_size": int(hf_config.sliding_window),
            "dsa_indexer_n_heads": int(hf_config.index_n_heads),
            "dsa_indexer_head_dim": int(hf_config.index_head_dim),
            "dsa_indexer_topk": int(hf_config.index_topk),
            # MLATransformerConfig defaults dsa_indexer_loss_coeff to None;
            # csa.py uses getattr(..., 0.0) but that fallback only fires when
            # the attribute is absent, not when it is explicitly None.  Provide
            # 0.0 here so callers that want a non-zero value can override via
            # set_extra_args (mirrors verl PR #6473 approach).
            "dsa_indexer_loss_coeff": 0.0,
            # --- mHC ---
            "enable_hyper_connections": True,
            "use_fused_mhc": True,
            "num_residual_streams": int(hf_config.hc_mult),
            "mhc_sinkhorn_iterations": int(hf_config.hc_sinkhorn_iters),
            # --- MoE ---
            "moe_grouped_gemm": True,
            "moe_router_pre_softmax": False,  # V4 uses post-topk normalisation
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_load_balancing_type": "noaux_tc",
            "moe_shared_expert_overlap": True,
            "moe_router_score_function": hf_config.scoring_func,
            "moe_router_enable_expert_bias": True,
            "moe_router_dtype": "fp32",
            "moe_permute_fusion": True,
            "moe_aux_loss_coeff": 0.0,
            "moe_router_topk": int(hf_config.num_experts_per_tok),
            "norm_topk_prob": bool(hf_config.norm_topk_prob),
            "moe_router_topk_scaling_factor": float(hf_config.routed_scaling_factor),
            "num_moe_experts": int(hf_config.n_routed_experts),
            "moe_ffn_hidden_size": int(hf_config.moe_intermediate_size),
            "moe_shared_expert_intermediate_size": int(
                hf_config.moe_intermediate_size * hf_config.n_shared_experts
            ),
            "moe_layer_freq": moe_layer_freq,
            # --- Hash routing & ClampedSwiGLU ---
            "moe_n_hash_layers": num_hash_layers,
            "actual_vocab_size": int(hf_config.vocab_size),
            "activation_func_clamp_value": float(hf_config.swiglu_limit),
            # --- Misc ---
            "gated_linear_unit": True,
            "bias_dropout_fusion": True,
            "persist_layer_norm": True,
            "masked_softmax_fusion": True,
            "attention_softmax_in_fp32": False,
            "hidden_dropout": 0.0,
        }
        base_config.update(mtp_args)

        # ``MLATransformerConfig`` is a dataclass — passing keys it does not
        # declare to ``__init__`` raises ``TypeError``. DSv4 brings a handful
        # of MCore-dev fields that are intentionally not part of the upstream
        # config dataclass yet (mirroring how NVIDIA-NeMo/Megatron-Bridge
        # assigns them post-init on a custom provider class). Split the
        # config: pass dataclass-known fields to ``__init__`` and ``setattr``
        # the rest on the resulting object so the downstream MCore modules
        # still see them via ``self.config.<field>``.
        known_fields = {f.name for f in dataclasses.fields(MLATransformerConfig)}
        init_kwargs = {k: v for k, v in base_config.items() if k in known_fields}
        extra_attrs = {k: v for k, v in base_config.items() if k not in known_fields}

        config = self._build_base_config(**init_kwargs)
        for attr, value in extra_attrs.items():
            setattr(config, attr, value)
        return config

    # ------------------------------------------------------------------
    # Model provider
    # ------------------------------------------------------------------

    def _get_gptmodel_args(self) -> dict:
        """Return GPTModel constructor arguments for DSv4."""
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=get_hf_rope_theta(self.hf_config),
        )

    def _get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        """Build the heterogeneous DSv4 block spec via the experimental builder.

        Unlike the standard ``get_gpt_decoder_block_spec`` used in LLMBridge,
        DSv4 needs the experimental attention variant block spec to produce
        the correct submodules for hash-MoE / CSA / HCA layers.

        The builder's keyword set has churned across MCore-dev commits
        (``use_transformer_engine`` was dropped, ``vp_stage`` was added); we
        introspect the current signature and pass only the kwargs it accepts.
        """
        builder = get_transformer_block_with_experimental_attention_variant_spec
        sig = inspect.signature(builder)
        accepted = set(sig.parameters)
        kwargs = {}
        if "vp_stage" in accepted and vp_stage is not None:
            kwargs["vp_stage"] = vp_stage
        if "use_transformer_engine" in accepted:
            kwargs["use_transformer_engine"] = True
        return builder(self.config, **kwargs)

    def _model_provider(
        self,
        post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]],
    ):
        """Return a ``provider(pre_process, post_process, vp_stage)`` callable."""
        share_embeddings_and_output_weights = getattr(
            self.hf_config, "tie_word_embeddings", False
        )

        def provider(pre_process, post_process, vp_stage: Optional[int] = None):
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            gptmodel_args = self._get_gptmodel_args()
            if vp_stage is not None and getattr(self, "has_vp_stage", False):
                gptmodel_args["vp_stage"] = vp_stage

            if self.config.mtp_num_layers and self.config.mtp_num_layers > 0:
                mtp_block_spec = get_gpt_mtp_block_spec(
                    self.config,
                    transformer_layer_spec,
                    use_transformer_engine=True,
                    vp_stage=vp_stage,
                )
                gptmodel_args["mtp_block_spec"] = mtp_block_spec

            model = GPTModel(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                **gptmodel_args,
            )
            for callback in post_model_creation_callbacks:
                callback(
                    model,
                    pre_process=pre_process,
                    post_process=post_process,
                    config=self.config,
                    hf_config=self.hf_config,
                )
            return model

        return provider

    # ------------------------------------------------------------------
    # Weight name mapping (partial coverage; see class docstring)
    # ------------------------------------------------------------------

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """Map MCore weight names to DeepSeek-V4 native HF checkpoint names.

        Coverage
        --------
        - Embedding / lm-head / final norm
        - MLA Q/KV projections, factored output projection (wo_a + wo_b)
        - Per-layer attention norms (attn_norm / kv_norm / q_norm)
        - MoE router, routed experts, shared expert
        - MTP transformer layers (via proxy substitution)

        Not yet covered (raise ``NotImplementedError``)
        -----------------------------------------------
        - mHC parameters (``hc_attn_fn`` / ``hc_attn_base`` / ``hc_attn_scale``,
          and the FFN/head variants). HC alpha scalars need a custom
          concatenation mapping that this minimal bridge does not provide.
        - DSA indexer weights and per-layer compressor weights.
        - Attention sink (TP-split).
        - Hash-routing ``tid2eid`` buffer.

        For full DSv4 checkpoint roundtrip, including FP8 / MXFP4
        dequantisation, use ``NVIDIA-NeMo/Megatron-Bridge`` instead.
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if mcore_weights_name.startswith("mtp.layers."):
            return self._convert_mtp_param(mcore_weights_name)

        if (
            "self_attention" in mcore_weights_name
            or "input_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)

        if "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)

        raise NotImplementedError(
            f"DSv4 bridge does not yet handle weight name: {mcore_weights_name!r}. "
            "Hyper-connection alphas, DSA indexer, compressor, attention sink, "
            "and tid2eid mappings are out of scope for this bridge; see "
            "NVIDIA-NeMo/Megatron-Bridge for the full implementation."
        )

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """Resolve MoE / shared-expert / FFN-norm weight names for DSv4."""
        # Layer index lives at position 2 in "decoder.layers.N.mlp...".
        layer_number = name.split(".")[2]
        convert_names: list[str] = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1].lstrip(".")
                    convert_names.extend(
                        [
                            template.format(
                                layer_number=layer_number, expert_id=expert_id
                            )
                            for template in mapping_names
                        ]
                    )
                else:
                    convert_names.extend(
                        [
                            template.format(layer_number=layer_number)
                            for template in mapping_names
                        ]
                    )
                break
        if not convert_names:
            raise NotImplementedError(f"Unsupported DSv4 MLP parameter name: {name!r}")
        return convert_names

    def _convert_mtp_param(self, name: str) -> list[str]:
        """Convert MTP transformer-layer params via a per-MTP proxy mapping.

        MTP transformer layers mirror decoder layer structure, so we substitute
        the ``mtp.layers.{i}.{transformer_layer|mtp_model_layer}.…`` prefix
        with a synthetic ``decoder.layers.{num_main + i}.…`` name and reuse
        the per-layer attention / MLP helpers. MTP-only norms / projections
        (``enorm``, ``hnorm``, ``eh_proj``, ``e_proj``, ``h_proj``,
        ``final_layernorm``) are not yet mapped and raise
        ``NotImplementedError``.
        """
        parts = name.split(".")
        # mtp.layers.{i}.<rest...>
        if len(parts) < 4 or parts[0] != "mtp" or parts[1] != "layers":
            raise NotImplementedError(f"Unexpected MTP parameter shape: {name!r}")
        mtp_idx = int(parts[2])
        inner_root = parts[3]
        if inner_root not in ("transformer_layer", "mtp_model_layer"):
            raise NotImplementedError(
                f"DSv4 MTP-specific parameter {name!r} not yet mapped. "
                "Implement once MTP checkpoint roundtrip is in scope."
            )
        n_main = int(self.hf_config.num_hidden_layers)
        global_idx = n_main + mtp_idx
        rest = ".".join(parts[4:])
        proxy_name = f"decoder.layers.{global_idx}.{rest}"
        if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
            return self._weight_name_mapping_attention(proxy_name)
        if "mlp" in proxy_name:
            return self._weight_name_mapping_mlp(proxy_name)
        raise NotImplementedError(f"Unsupported DSv4 MTP inner parameter: {name!r}")
