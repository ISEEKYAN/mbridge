# Adapted from NVIDIA Megatron-Bridge

import logging
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import torch
from mbridge.peft.adapter_wrapper import AdapterWrapper
from mbridge.peft.base import PEFT
from mbridge.peft.lora_layers import (LinearAdapter, LoRAGroupedLinear,
                                      LoRALinear, LoRATopKRouter)
from mbridge.peft.module_matcher import ModuleMatcher
from mbridge.peft.utils import (ParallelLinearAdapter,
                                get_adapter_attributes_from_linear,
                                is_expert_linear)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.moe.router import TopKRouter
from torch import nn

logger = logging.getLogger(__name__)


def _should_treat_linear_fc1_as_unfused(full_name: str) -> bool:
    """Return True when CanonicalLoRA should keep linear_fc1 as a single adapter."""

    return full_name.startswith("vision_model.") or full_name.endswith(".mlp.experts.linear_fc1")


class ModuleDict(nn.ModuleDict):
    """
    nn.ModuleDict with a sharded_state_dict implementation for checkpointing
    """

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> "ShardedStateDict":
        """Retrieve the sharded state dictionary of the wrapped module and adapter.

        This method is used for distributed checkpointing, combining the sharded states
        of both the main module and the adapter.

        Args:
            prefix (str): A prefix added to parameter and buffer names. Defaults to ''.
            sharded_offsets (Tuple[Tuple[int, int, int]]): Offsets for sharded parameters.
                                                           Defaults to an empty tuple.
            metadata (Optional[dict]): Additional metadata for the sharded state.
                                       Defaults to None.

        Returns:
            ShardedStateDict: The combined sharded state dictionary.
        """
        sharded_state_dict = {}
        for key, layer in self.items():
            sharded_state_dict.update(layer.sharded_state_dict(f"{prefix}{key}.", sharded_offsets, metadata))
        return sharded_state_dict


class LoRALinearSplitQKV(AdapterWrapper):
    """An adapter wrapper for `linear_qkv` where q, k, v are three separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def _interleave_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Interleave QKV outputs to match Megatron's packed ordering."""

        config = self.to_wrap.config
        head_num = getattr(config, "num_attention_heads", None)
        num_query_groups = getattr(config, "num_query_groups", None)
        head_size = getattr(config, "kv_channels", None)

        if head_size is None:
            hidden_size = getattr(config, "hidden_size", None)
            if head_num is not None and hidden_size is not None:
                head_size = hidden_size // head_num
            elif num_query_groups:
                if key.size(-1) % num_query_groups != 0:
                    raise ValueError("Key projection size must be divisible by num_query_groups.")
                head_size = key.size(-1) // num_query_groups
            elif head_num is not None:
                if query.size(-1) % head_num != 0:
                    raise ValueError("Query projection size must be divisible by num_attention_heads.")
                head_size = query.size(-1) // head_num
            else:
                raise ValueError(
                    "Cannot infer head size without kv_channels or hidden_size/num_attention_heads or num_query_groups."
                )

        if head_num is None:
            if query.size(-1) % head_size != 0:
                raise ValueError("Query projection size must be divisible by head_size.")
            head_num = query.size(-1) // head_size

        if not num_query_groups:
            if key.size(-1) % head_size != 0:
                raise ValueError("Key projection size must be divisible by head_size.")
            num_query_groups = key.size(-1) // head_size

        if head_num % num_query_groups != 0:
            raise ValueError("num_attention_heads must be divisible by num_query_groups.")

        heads_per_group = head_num // num_query_groups

        leading_shape = query.shape[:-1]
        query = query.reshape(-1, head_num, head_size)
        key = key.reshape(-1, num_query_groups, head_size)
        value = value.reshape(-1, num_query_groups, head_size)

        output_gate = getattr(config, "attention_output_gate", False)

        qkv_chunks = []
        for i in range(num_query_groups):
            q_group = query[:, i * heads_per_group : (i + 1) * heads_per_group, :]
            k_group = key[:, i : i + 1, :]
            v_group = value[:, i : i + 1, :]
            qkv_chunks.append(q_group)
            if output_gate:
                qkv_chunks.append(torch.zeros_like(q_group))
            qkv_chunks.append(k_group)
            qkv_chunks.append(v_group)

        qkv = torch.cat(qkv_chunks, dim=1)
        return qkv.reshape(*leading_shape, -1)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias

        # Only compute adapters that exist (not None)
        query = self.adapter.adapter_q(layernorm_output) if self.adapter.adapter_q is not None else None
        key = self.adapter.adapter_k(layernorm_output) if self.adapter.adapter_k is not None else None
        value = self.adapter.adapter_v(layernorm_output) if self.adapter.adapter_v is not None else None

        if query is None and key is None and value is None:
            return linear_output, bias

        # For canonical LoRA, if only a subset of Q/K/V adapters are present,
        # fall back to a simpler concatenation (Q then K then V) which matches
        # the Megatron QKV weight order for standard (non-GQA-interleaved) layouts.
        if query is None:
            adapter_output = torch.cat(
                [t for t in [key, value] if t is not None], dim=-1,
            )
        elif key is None and value is None:
            adapter_output = query
        elif key is None:
            adapter_output = torch.cat([query, value], dim=-1)
        elif value is None:
            adapter_output = torch.cat([query, key], dim=-1)
        else:
            adapter_output = self._interleave_qkv(query, key, value)

        return linear_output + adapter_output, bias


class LoRALinearSplitFC1UpGate(AdapterWrapper):
    """An adapter wrapper for `linear_fc1` where up_proj and gate_proj are two separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias
        adapter_parts = []
        if self.adapter.adapter_gate is not None:
            adapter_parts.append(self.adapter.adapter_gate(layernorm_output))
        if self.adapter.adapter_up is not None:
            adapter_parts.append(self.adapter.adapter_up(layernorm_output))
        if not adapter_parts:
            return linear_output, bias
        adapter_output = torch.cat(adapter_parts, dim=-1)
        return linear_output + adapter_output, bias


@dataclass
class CanonicalLoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.
    Canonical LoRA applies LoRA on Q, K, V projection matrices separately, as well as Up and Gate projection
    matrices separately. This follows more closely with Huggingface's implementation of LoRA.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_q', 'linear_k', 'linear_v', 'linear_proj',
                                           'linear_fc1_up', 'linear_fc1_gate', 'linear_fc2'].
                - 'linear_q', 'linear_k', 'linear_v': Apply LoRA to the linear layer used for query, key, and value
                        projections in self-attention. This is fused into one matrix in LoRA, but left as three
                        separate matrices in Canonical LoRA.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1_up', 'linear_fc1_gate': Apply LoRA to the Up proj and Gate proj layers.
                        These two together constitute the first fully-connected layer in MLP in LoRA.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_q', '*.layers.1.*.linear_q'] to add LoRA to only linear_q
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRA to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        lora_A_init_method (str): Initialization method for LoRA A matrix. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for LoRA B matrix. Defaults to "zero".
    """

    target_modules: List[str] = field(
        default_factory=lambda: [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"

    def __post_init__(self) -> None:
        """
        Initialize the canonical mapping and call the parent post_init.

        Construct a mapping from the target module as supported in LoRA() to the specific parts of the layer for which
        adapter is applied.

        For example, if user specifies target_module = ['linear_q', 'linear_k', 'linear_proj', 'linear_fc1_up'], then
        canonical_lora_mapping = {
            "linear_qkv": {'linear_q', 'linear_k'},
            "linear_proj": {'linear_proj'},  # the value of this key does not matter
            "linear_fc1": {'linear_fc1_up'},
        }

        If user specifies target_module = ['*.layers.0.*.linear_q', '*.layers.1.*.linear_q'], then
        canonical_lora_mapping = {
            "'*.layers.0.*.linear_qkv'": {'linear_q'},
            "'*.layers.1.*.linear_qkv'": {'linear_q'},
        }

        """
        for target in self.target_modules:
            assert not target.endswith("linear_qkv"), (
                "Canonical LoRA does not support target 'linear_qkv'. Either use 'linear_qkv' with LoRA() or "
                "use ['linear_q', 'linear_k', 'linear_v'] with Canonical LoRA"
            )
            assert not target.endswith("linear_fc1"), (
                "Canonical LoRA does not support target 'linear_fc1'. Either use 'linear_fc1' with LoRA() or "
                "use ['linear_fc1_up', 'linear_fc1_gate'] with Canonical LoRA"
            )

            if target.endswith("linear_q"):
                self.canonical_mapping[target.replace("linear_q", "linear_qkv")].add("linear_q")
            elif target.endswith("linear_k"):
                self.canonical_mapping[target.replace("linear_k", "linear_qkv")].add("linear_k")
            elif target.endswith("linear_v"):
                self.canonical_mapping[target.replace("linear_v", "linear_qkv")].add("linear_v")
            elif target.endswith("linear_fc1_up"):
                self.canonical_mapping[target.replace("linear_fc1_up", "linear_fc1")].add("linear_fc1_up")
            elif target.endswith("linear_fc1_gate"):
                self.canonical_mapping[target.replace("linear_fc1_gate", "linear_fc1")].add("linear_fc1_gate")
            else:
                self.canonical_mapping[target].add(target)

    def transform(self, m: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (Optional[str]): Name of the module (if applicable). Defaults to None.
            prefix (Optional[str]): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """

        # Skip already transformed modules
        if isinstance(m, (LinearAdapter, LoRALinear, LoRAGroupedLinear, LoRALinearSplitQKV, LoRALinearSplitFC1UpGate, LoRATopKRouter)):
            return m

        if (ans := self.match(m, name, prefix)) is not None:
            (match, full_name) = ans
            if isinstance(m, nn.Linear):
                return LinearAdapter(
                    m, dim=self.dim, alpha=self.alpha, dropout=self.dropout, lora_A_init_method=self.lora_A_init_method
                )

            from megatron.core.tensor_parallel import (
                ColumnParallelLinear, RowParallelLinear,
            )
            from mbridge.peft.utils import TECL, TERL
            _supported_types = (
                ColumnParallelLinear, RowParallelLinear, TopKRouter,
            ) + TECL + TERL
            if not isinstance(m, _supported_types):
                logger.warning(
                    f"CanonicalLoRA target pattern matched module '{full_name}' "
                    f"of type {type(m).__name__}, but this type is not supported "
                    f"for LoRA adaptation. Skipping."
                )
                return m

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(m, is_expert=is_expert)

            adapter_kwargs = dict(
                dim=self.dim,
                base_linear_name=full_name,
                activation="identity",
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
                is_expert=is_expert,
                disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )

            # Per-expert LoRA: each expert gets its own adapter
            num_gemms = getattr(m, "num_gemms", 0)
            if is_expert and num_gemms > 0:
                logger.info(
                    f"Adding per-expert lora to: {full_name} "
                    f"(num_local_experts={num_gemms})"
                )
                adapters = nn.ModuleList()
                for i in range(num_gemms):
                    adapters.append(
                        ParallelLinearAdapter(
                            attrs.in_features, attrs.out_features,
                            **{**adapter_kwargs, "base_linear_name": f"{full_name}.expert{i}"},
                        )
                    )
                return LoRAGroupedLinear(m, adapters)

            if name == "linear_fc1" and _should_treat_linear_fc1_as_unfused(full_name):
                logger.info(f"Adding lora to: {full_name} (treating unsupported canonical linear_fc1 as unfused)")
                adapter = ParallelLinearAdapter(attrs.in_features, attrs.out_features, **adapter_kwargs)
                return LoRALinear(m, adapter)

            canonical_submodules = self.canonical_mapping[match]
            logger.info(f"Adding lora to: {full_name} ({canonical_submodules})")
            if name == "linear_qkv":
                adapter_q, adapter_k, adapter_v = None, None, None
                kv_out_features = m.config.kv_channels * m.config.num_query_groups
                q_out_features = m.config.kv_channels * m.config.num_attention_heads
                if "linear_q" in canonical_submodules:
                    adapter_q = ParallelLinearAdapter(attrs.in_features, q_out_features, **adapter_kwargs)
                if "linear_k" in canonical_submodules:
                    adapter_k = ParallelLinearAdapter(attrs.in_features, kv_out_features, **adapter_kwargs)
                if "linear_v" in canonical_submodules:
                    adapter_v = ParallelLinearAdapter(attrs.in_features, kv_out_features, **adapter_kwargs)
                adapters = ModuleDict({"adapter_q": adapter_q, "adapter_k": adapter_k, "adapter_v": adapter_v})
                return LoRALinearSplitQKV(m, adapters)

            if name == "linear_fc1":
                stride = getattr(m, 'stride', 1)
                if stride <= 1:
                    # Non-GLU: single adapter with full out_features.
                    # When gated_linear_unit=False, linear_fc1 has no gate/up split
                    # and the canonical target_modules 'linear_fc1_up'/'linear_fc1_gate'
                    # are treated as targeting the sole weight matrix.
                    logger.info(
                        f"Adding lora to: {full_name} (non-gated, single adapter, canonical_submodules={canonical_submodules})"
                    )
                    adapter = ParallelLinearAdapter(attrs.in_features, attrs.out_features, **adapter_kwargs)
                    return LoRALinear(m, adapter)
                # GLU: split gate/up, each with out_features // 2
                adapter_up, adapter_gate = None, None
                if "linear_fc1_up" in canonical_submodules:
                    adapter_up = ParallelLinearAdapter(attrs.in_features, attrs.out_features // 2, **adapter_kwargs)
                if "linear_fc1_gate" in canonical_submodules:
                    adapter_gate = ParallelLinearAdapter(attrs.in_features, attrs.out_features // 2, **adapter_kwargs)
                adapters = ModuleDict({"adapter_up": adapter_up, "adapter_gate": adapter_gate})
                return LoRALinearSplitFC1UpGate(m, adapters)

            adapter = ParallelLinearAdapter(attrs.in_features, attrs.out_features, **adapter_kwargs)
            logger.info(f"Adding lora to: {full_name}")
            if isinstance(m, TopKRouter):
                return LoRATopKRouter(m, adapter)
            return LoRALinear(m, adapter)

        return m
