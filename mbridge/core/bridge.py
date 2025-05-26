# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from typing import Generator

import torch
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from torch.nn import functional as F
from transformers import AutoConfig

from .parallel_states import ParallelStates
from .util import get_model, load_some_hf_weight, unwrap_model


class BaseBridge(ABC):
    """
    Base abstract class for all bridge implementations.

    This class defines the interface that all bridge classes must implement.
    A bridge connects Hugging Face model configurations and weights with
    Megatron-Core's optimized implementations.
    """

    @abstractmethod
    def get_model(self, weight_path: str = None, **kwargs):
        """
        Get a model instance with optional weight loading.

        Args:
            weight_path: Path to model weights
            **kwargs: Additional arguments for model creation

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def load_weights(self, models: list[torch.nn.Module], weights_path: str) -> None:
        """
        Load weights from a Hugging Face model into a Megatron-Core model.

        Args:
            models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)
            weights_path: Path to the weights file
        """
        pass

    @abstractmethod
    def set_extra_args(self, **kwargs):
        """
        Set additional configuration arguments.

        Args:
            **kwargs: Key-value pairs of additional arguments
        """
        pass

    @abstractmethod
    def export_weights(
        self, models: list[torch.nn.Module]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Export weights from a Megatron-Core model to a Hugging Face model.

        Args:
            models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)

        Returns:
            Generator of tuples containing (weight name, weight tensor)
            The whole weights (no TP/PP/EP split) are in the generator
        """
        pass


class Bridge(BaseBridge):
    """
    Base model bridge class.

    This class implements the core functionality to bridge between
    Hugging Face models and Megatron-Core optimized implementations.
    """

    def __init__(
        self,
        hf_config: AutoConfig,
        dtype: torch.dtype = torch.bfloat16,
        parallel_states: ParallelStates = None,
    ):
        """
        Initialize a bridge instance.

        Args:
            hf_config: Hugging Face model configuration
            dtype: Data type for model parameters
            parallel_states: Parallel processing states, or None to use default
        """
        self.hf_config = hf_config
        self.extra_args = {}
        self.dtype = dtype
        self.mpu = parallel_states
        if self.mpu is None:
            self.mpu = ParallelStates.get_parallel_state()
        self.config = self._build_config()

    def get_model(
        self,
        weight_path: str = None,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        fp16: bool = False,
        bf16: bool = True,
        encoder_pipeline_model_parallel_size: int = 0,
        use_torch_fsdp2: bool = False,
        use_custom_fsdp: bool = False,
        use_precision_aware_optimizer: bool = False,
        use_cpu_initialization: bool = False,
        init_model_with_meta_device: bool = False,
        overlap_param_gather_with_optimizer_step: bool = False,
        data_parallel_random_init: bool = True,
        optimizer_config: dict = None,
        value_model: bool = False,
        extra_provider_args: dict = {},
        **kwargs,
    ):
        """
        Get a model instance.

        Args:
            weight_path: Path to model weights
            model_type: Type of model to create
            wrap_with_ddp: Whether to wrap with DDP
            fp16: Whether to use FP16 precision
            bf16: Whether to use BF16 precision
            encoder_pipeline_model_parallel_size: Size of encoder pipeline parallelism
            use_torch_fsdp2: Whether to use PyTorch FSDP 2.0
            use_custom_fsdp: Whether to use custom FSDP
            use_precision_aware_optimizer: Whether to use precision-aware optimizer
            use_cpu_initialization: Whether to initialize on CPU
            init_model_with_meta_device: Whether to initialize with meta device
            overlap_param_gather_with_optimizer_step: Whether to overlap parameter gathering
            data_parallel_random_init: Whether to use random initialization in data parallel
            optimizer_config: Optimizer configuration
            value_model: Whether to create a value model
            extra_provider_args: Additional arguments for the model provider
            **kwargs: Additional arguments

        Returns:
            Model instance
        """
        share_embeddings_and_output_weights = getattr(
            self.hf_config, "tie_word_embeddings", False
        )
        if (
            share_embeddings_and_output_weights
            and self.mpu.vpp_size
            and self.mpu.vpp_size > 1
        ):
            raise ValueError("tie_word_embeddings is not supported for VPP > 1")
        model = get_model(
            self._model_provider(
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                value_model=value_model,
                **extra_provider_args,
            ),
            model_type=model_type,
            wrap_with_ddp=wrap_with_ddp,
            fp16=fp16,
            bf16=bf16,
            virtual_pipeline_model_parallel_size=self.mpu.vpp_size,
            encoder_pipeline_model_parallel_size=encoder_pipeline_model_parallel_size,
            use_torch_fsdp2=use_torch_fsdp2,
            use_custom_fsdp=use_custom_fsdp,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
            use_cpu_initialization=use_cpu_initialization,
            init_model_with_meta_device=init_model_with_meta_device,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            data_parallel_random_init=data_parallel_random_init,
            optimizer_config=optimizer_config,
            **kwargs,
        )
        if weight_path:
            self.load_weights(model, weight_path)
        return model

    def load_weights(self, models: list[torch.nn.Module], weights_path: str) -> None:
        """
        Load weights from a Hugging Face model into a Megatron-Core model.

        Args:
            models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)
            weights_path: Path to the weights file
        """
        for i, model in enumerate(models):
            # map local weight names to global weight names
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            # map local weight names to huggingface weight names
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }
            # only tp_rank0/etp_rank0 load from disk, others load from tp_rank0/etp_rank0
            to_load_from_disk = []
            for local_name, hf_names in local_to_hf_map.items():
                if ".mlp.experts.linear_fc" in local_name:
                    if self.mpu.etp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.extend(hf_names)

            # load huggingface weights
            hf_weights_map = load_some_hf_weight(weights_path, to_load_from_disk)
            # import mcore weights
            for local_name, hf_names in local_to_hf_map.items():
                param = model.state_dict()[local_name]
                # hf format to mcore format
                if set(to_load_from_disk) & set(hf_names):
                    hf_weights = [hf_weights_map[x] for x in hf_names]
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None
                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in local_name:
                    # split mcore weights across etp
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device) for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    # split mcore weights across tp
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.tp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device) for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                # load
                param.copy_(param_to_load)

    def set_extra_args(self, **kwargs):
        """
        Set additional configuration arguments.

        Args:
            **kwargs: Key-value pairs of additional arguments
        """
        self.extra_args.update(kwargs)
        self.config = self._build_config()

    def _build_base_config(self, **kwargs):
        """
        Build the base configuration for the model.

        Args:
            **kwargs: Additional configuration overrides

        Returns:
            TransformerConfig: The constructed transformer configuration
        """
        hf_config = self.hf_config
        dtype = self.dtype
        base_config = {
            # Model architecture parameters
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_query_groups": hf_config.num_key_value_heads,
            "ffn_hidden_size": hf_config.intermediate_size,
            "attention_dropout": hf_config.attention_dropout,
            "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
            "kv_channels": getattr(hf_config, "head_dim", None),
            "layernorm_epsilon": hf_config.rms_norm_eps,
            # Activation and normalization
            "activation_func": F.silu,
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
            # Data types
            "pipeline_dtype": dtype,
            "params_dtype": dtype,
            "bf16": dtype is torch.bfloat16,
            # Parallel configuration
            "tensor_model_parallel_size": self.mpu.tp_size,
            "pipeline_model_parallel_size": self.mpu.pp_size,
            "expert_model_parallel_size": self.mpu.ep_size,
            "expert_tensor_parallel_size": self.mpu.etp_size,
            "virtual_pipeline_model_parallel_size": self.mpu.vpp_size,
            "context_parallel_size": self.mpu.cp_size,
            "sequence_parallel": self.mpu.tp_size > 1,
            # Common settings
            "variable_seq_lengths": True,
            "masked_softmax_fusion": True,
            "moe_token_dispatcher_type": "alltoall",
        }

        # Update with any provided overrides
        base_config.update(kwargs)
        # print(f"Overridden TF init config: {base_config}")

        return TransformerConfig(**base_config)

    def _build_config(self):
        """
        Build the configuration for the model.
        This method must be implemented by subclasses.

        Returns:
            Configuration object for the model

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _model_provider(
        self, share_embeddings_and_output_weights=False, value_model=False
    ):
        """
        Create a model provider function.
        This method must be implemented by subclasses.

        Args:
            share_embeddings_and_output_weights: Whether to share embedding weights
            value_model: Whether to create a value model

        Returns:
            Function that provides a model

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """

        def provider(pre_process, post_process):
            raise NotImplementedError("Subclasses must implement this method")
            model = None
            return model

        return provider

    def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module
    ) -> dict[str, str]:
        """
        Map local weight names to global weight names, supporting VPP and EP.

        Args:
            model: The model instance

        Returns:
            dict: Mapping from local weight names to global weight names
        """
        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "decoder"):
            for idx, layer in enumerate(model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [
            k for k in model.state_dict().keys() if "_extra_state" not in k
        ]
        ret = {}
        for param_name in all_param_names:
            keyword = "decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(
                    f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
                )
            else:
                ret[param_name] = param_name
        # ep
        if self.mpu.ep_size > 1:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            local_expert_to_global_expert = {
                i: i + num_experts_per_rank * self.mpu.ep_rank
                for i in range(num_experts_per_rank)
            }
            for k in ret.keys():
                if ".mlp.experts.linear_fc" in k:
                    name_prefix, local_expert_id = k.split(".weight")
                    global_expert_idx = local_expert_to_global_expert[
                        int(local_expert_id)
                    ]
                    ret[k] = f"{name_prefix}.weight{global_expert_idx}"

        return ret

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    def _weight_name_mapping_attention(self, name: str) -> str:
        """
        Map attention weight names from MCore to Hugging Face.

        Args:
            name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._ATTENTION_MAPPING.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number) for x in mapping_names]
                )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    def _weight_name_mapping_mlp(self, name: str) -> str:
        """
        Map MLP weight names from MCore to Hugging Face.

        Args:
            name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number) for x in mapping_names]
                )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if mcore_weights_name in direct_name_mapping:
            return [direct_name_mapping[mcore_weights_name]]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        Export MCore weights to Hugging Face format.

        Takes MCore weight names and tensor, outputs Hugging Face weight names and tensors.
        Due to MCore's runtime optimizations involving weight merging, output can be a list.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor

        Returns:
            tuple: (hf_names, hf_weights) - lists of Hugging Face weight names and tensors

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        if len(hf_names) == 1:
            return [hf_names[0]], [mcore_weights]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # split qkv
            assert len(hf_names) == 3
            # split qkv
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            head_dim = getattr(
                self.hf_config, "head_dim", hidden_dim // num_attention_heads
            )
            out_shape = (
                [num_key_value_heads, -1, hidden_dim]
                if ".bias" not in mcore_weights_name
                else [num_key_value_heads, -1]
            )
            qkv = mcore_weights.view(*out_shape)
            q_len = head_dim * num_attention_heads // num_key_value_heads
            k_len = head_dim
            v_len = head_dim
            single_out_shape = (
                [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
            )
            q = qkv[:, :q_len].reshape(*single_out_shape)
            k = qkv[:, q_len : q_len + k_len].reshape(*single_out_shape)
            v = qkv[:, q_len + k_len :].reshape(*single_out_shape)
            return hf_names, [q, k, v]

        elif "linear_fc1.weight" in mcore_weights_name:
            # split gate_proj and up_proj
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Import Hugging Face weights to MCore format.

        Takes Hugging Face weight names and tensors, outputs MCore weight tensor.
        Due to MCore's runtime optimizations involving weight merging, input is a list.

        Args:
            mcore_weights_name: MCore weight name
            hf_weights: List of Hugging Face weight tensors

        Returns:
            torch.Tensor: MCore weight tensor

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        if len(hf_weights) == 1:
            return hf_weights[0]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            head_dim = getattr(
                self.hf_config, "head_dim", hidden_dim // num_attention_heads
            )
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // group_dim
            q = q.view(
                [
                    real_num_key_value_heads,
                    group_dim,
                    -1,
                ]
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qkv
        elif "linear_fc1.weight" in mcore_weights_name:
            # merge gate_proj and up_proj
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_merge_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: list[torch.Tensor],
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge weights across tensor parallel ranks.
        In mcore format

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: List of MCore weight tensors from different TP ranks
            param: Parameter tensor

        Returns:
            torch.Tensor: Merged weight tensor
        """
        if self.mpu.tp_size == 1:
            assert len(mcore_weights) == 1
            return mcore_weights[0]
        if "mlp.experts.linear_fc" in mcore_weights_name:
            assert len(mcore_weights) == self.mpu.etp_size
        else:
            assert len(mcore_weights) == self.mpu.tp_size
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            return torch.cat(mcore_weights, dim=0)
        elif "linear_fc1.weight" in mcore_weights_name:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for mcore_weight in mcore_weights:
                gate, up = mcore_weight.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            ret = torch.cat((gate, up), dim=0)

        elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
            ret = torch.cat(mcore_weights, dim=1)
        else:
            assert (
                hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
            )
            ret = torch.cat(mcore_weights, dim=param.partition_dim)

        return ret

    def _weight_split_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: torch.Tensor,
        param: torch.Tensor,
        tp_split_size: int,
    ) -> list[torch.Tensor]:
        """
        Split weight tensor across tensor parallel ranks.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor
            param: Parameter tensor

        Returns:
            list: List of weight tensors split for each TP rank
        """
        if tp_split_size == 1:
            return [mcore_weights]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            return mcore_weights.chunk(tp_split_size)
        elif "linear_fc1.weight" in mcore_weights_name:
            _, [gate, up] = self._weight_to_hf_format(mcore_weights_name, mcore_weights)
            gates = gate.chunk(tp_split_size)
            ups = up.chunk(tp_split_size)
            ret = [
                self._weight_to_mcore_format(mcore_weights_name, [g, u])
                for g, u in zip(gates, ups)
            ]
        elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
            ret = mcore_weights.chunk(tp_split_size, dim=1)
        else:
            if param.shape == mcore_weights.shape:
                return [mcore_weights for _ in range(tp_split_size)]
            assert len(param.shape) == len(mcore_weights.shape)
            for partition_dim, (s1, s2) in enumerate(
                zip(param.shape, mcore_weights.shape)
            ):
                if s1 != s2:
                    break

            ret = mcore_weights.chunk(tp_split_size, dim=partition_dim)
        return ret


# Model registry
_MODEL_REGISTRY = {}


def register_model(model_types):
    """
    Model registration decorator.

    Args:
        model_types: String or list of strings representing model type identifiers

    Returns:
        Decorator function
    """
    if isinstance(model_types, str):
        model_types = [model_types]

    def decorator(cls):
        for model_type in model_types:
            _MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator
