import inspect
from copy import deepcopy
from typing import Callable, Optional

import torch

from mbridge.core import VLMBridge
from mbridge.core.util import unwrap_model
from mbridge.models.qwen3_5.model import Qwen3p5VLModel
from mbridge.models.qwen3_5.qwen3_5_safetensor import Qwen3p5SafeTensorIO


class Qwen3p5VlBaseBridge(VLMBridge):

    def _get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        """
        Gets the transformer layer specification.

        Creates and returns a specification for the transformer layers based on
        the current configuration.

        Returns:
            TransformerLayerSpec: Specification for transformer layers

        Raises:
            AssertionError: If normalization is not RMSNorm
        """
        assert (
            self.config.normalization == "RMSNorm"
        ), "only RMSNorm is supported for now"

        try:
            from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
                is_linear_attention_variant,
                get_transformer_block_with_experimental_attention_variant_spec,
            )
        except ImportError:
            is_linear_attention_variant = None
            get_transformer_block_with_experimental_attention_variant_spec = None

        if is_linear_attention_variant is not None and is_linear_attention_variant(getattr(self.config, "experimental_attention_variant", None)):
            # check if get_transformer_block_with_experimental_attention_variant_spec has vp_stage parameter
            sig = inspect.signature(get_transformer_block_with_experimental_attention_variant_spec)
            self.has_vp_stage = "vp_stage" in sig.parameters  # for mcore 0.12 compatibility
            extra_args = {}
            if self.has_vp_stage:
                extra_args["vp_stage"] = vp_stage

            # Use experimental attention variant spec for linear attention (e.g., gated_delta_net)
            transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
                self.config,
                **extra_args,
            )
        else:
            raise ImportError("experimental_attention_variant is not supported, please megatron-lm dev branch")

        return transformer_layer_spec

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config.text_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["language_model.output_layer.weight"] = (
                "model.language_model.embed_tokens.weight"
            )

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config.text_config, "tie_word_embeddings", False):
            return ["model.language_model.embed_tokens.weight"]
        return []

    def _get_mcore_config_by_name(self, mcore_weights_name: str):
        return self.config

    def _get_safetensor_io(self, weights_path: str):
        #TODO: 暂时先不处理 mtp 层
        return Qwen3p5SafeTensorIO(self._get_actual_hf_path(weights_path), ignore_mtp=True)

    def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module, consider_ep: bool = True
    ) -> dict[str, str]:
        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "language_model") and hasattr(
            model.language_model, "decoder"
        ):
            for idx, layer in enumerate(model.language_model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [
            k for k in model.state_dict().keys() if "_extra_state" not in k
        ]
        ret = {}
        for param_name in all_param_names:
            keyword = "language_model.decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(
                    f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
                )
            else:
                ret[param_name] = param_name

        # ep
        if self.mpu.ep_size > 1 and consider_ep:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            local_expert_to_global_expert = {
                i: i + num_experts_per_rank * self.mpu.ep_rank
                for i in range(num_experts_per_rank)
            }
            for k in ret.keys():
                v = ret[k]
                if ".mlp.experts.linear_fc" in v:
                    name_prefix, local_expert_id = v.split(".weight")
                    global_expert_idx = local_expert_to_global_expert[
                        int(local_expert_id)
                    ]
                    ret[k] = f"{name_prefix}.weight{global_expert_idx}"

        return ret

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._ATTENTION_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._MLP_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_other(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._OTHER_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_visual(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[2]
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._VISUAL_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
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

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "vision_model" in mcore_weights_name:
            return self._weight_name_mapping_visual(mcore_weights_name)

        if ".self_attention." in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name} {self._DIRECT_MAPPING}"
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

        self_attn_output_gate = getattr(self.config, "attention_output_gate", False)

        if len(hf_names) == 1:
            # pad embeding and output layer
            if self.make_vocab_size_divisible_by is not None and (
                "embedding.word_embeddings.weight" in mcore_weights_name
                or "output_layer.weight" in mcore_weights_name
            ):
                assert mcore_weights.shape[0] == self.padded_vocab_size
                assert self.vocab_size is not None

                return [hf_names[0]], [mcore_weights[: self.vocab_size]]

            # moe
            if ".mlp.experts.linear_fc" in mcore_weights_name:
                # get export index
                experts_key = hf_names[0]
                experts_idx = int(mcore_weights_name.split(".weight")[-1])

                if experts_key not in self.export_weights_buff:
                    self.export_weights_buff[experts_key] = {}
                assert experts_idx not in self.export_weights_buff
                self.export_weights_buff[experts_key][experts_idx] = mcore_weights

                if (
                    len(self.export_weights_buff[experts_key])
                    < self.config.num_moe_experts
                ):
                    return [], []

                mcore_weights_list = []
                for idx in range(self.config.num_moe_experts):
                    mcore_weights_list.append(
                        self.export_weights_buff[experts_key].pop(idx)
                    )
                self.export_weights_buff.pop(experts_key)
                return [hf_names[0]], [torch.stack(mcore_weights_list)]
            elif "self_attention.out_norm.weight" in mcore_weights_name:
                return [hf_names[0]], [mcore_weights + 1]

            return [hf_names[0]], [mcore_weights]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            assert "vision_model" not in mcore_weights_name
            # split qkv
            assert len(hf_names) == 3
            # split qkv
            num_key_value_heads = self.hf_config.text_config.num_key_value_heads
            hidden_dim = self.hf_config.text_config.hidden_size
            num_attention_heads = self.hf_config.text_config.num_attention_heads

            head_dim = getattr(
                self.hf_config.text_config,
                "head_dim",
                hidden_dim // num_attention_heads,
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
            g = None
            if self_attn_output_gate:
                g = qkv[:, q_len : q_len + q_len].reshape(*single_out_shape)
                q_len += q_len

            k = qkv[:, q_len : q_len + k_len].reshape(*single_out_shape)
            v = qkv[:, q_len + k_len :].reshape(*single_out_shape)

            if self_attn_output_gate:
                _out_shape = (
                    [num_attention_heads, -1, hidden_dim]
                    if ".bias" not in mcore_weights_name
                    else [num_attention_heads, -1]
                )
                q = q.view(_out_shape)
                g = g.view(_out_shape)

                q = torch.cat([q, g], dim=1).view(*single_out_shape).contiguous()

            return hf_names, [q, k, v]

        elif (
            "vision_model" not in mcore_weights_name and (
                "linear_fc1.weight" in mcore_weights_name
                or "linear_fc1.bias" in mcore_weights_name
            )
        ):
            # split gate_proj and up_proj
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]

        elif "self_attention.in_proj.weight" in mcore_weights_name:
            assert len(hf_names) == 4
            hidden_size = self.hf_config.text_config.hidden_size
            linear_num_key_heads = self.hf_config.text_config.linear_num_key_heads
            linear_key_head_dim = self.hf_config.text_config.linear_key_head_dim
            linear_num_value_heads = self.hf_config.text_config.linear_num_value_heads
            linear_value_head_dim = self.hf_config.text_config.linear_value_head_dim

            k_dim = linear_num_key_heads * linear_key_head_dim
            v_dim = linear_num_value_heads * linear_value_head_dim
            split_shape = [
                k_dim,
                k_dim,
                v_dim,
                v_dim,
                linear_num_value_heads,
                linear_num_value_heads,
            ]
            weight_lst = mcore_weights.split(split_shape, dim=0)
            # weight_lst: [wq, wk, wv, wz, wb, wa]
            assert len(weight_lst) == 6
            wq, wk, wv, wz, wb, wa = weight_lst

            # qk_shape = [linear_num_key_heads, linear_key_head_dim, -1]
            # vz_shape = [linear_num_key_heads, v_dim // linear_num_key_heads, -1]
            # ba_shape = [linear_num_key_heads, linear_num_value_heads // linear_num_key_heads, -1]
            wq = wq.view([-1, hidden_size])
            wk = wk.view([-1, hidden_size])
            wv = wv.view([-1, hidden_size])

            wz = wz.view([-1, hidden_size]).contiguous()
            wb = wb.view([-1, hidden_size]).contiguous()
            wa = wa.view([-1, hidden_size]).contiguous()

            qkv_weight = torch.cat([wq, wk, wv], dim=0).contiguous()

            return hf_names, [qkv_weight, wz, wb, wa]

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
        self_attn_output_gate = getattr(self.config, "attention_output_gate", False)

        if len(hf_weights) == 1:
            # vision model
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

            # pad embeding and output layer
            if self.make_vocab_size_divisible_by is not None and (
                "embedding.word_embeddings.weight" in mcore_weights_name
                or "output_layer.weight" in mcore_weights_name
            ):
                assert hf_weights[0].shape[0] == self.vocab_size
                assert self.padded_vocab_size is not None

                embed_dim = hf_weights[0].shape[1]
                extra_zeros = torch.zeros(
                    (self.padded_vocab_size - self.vocab_size, embed_dim),
                    device=hf_weights[0].device,
                    dtype=hf_weights[0].dtype,
                )
                return torch.cat((hf_weights[0], extra_zeros), dim=0)

            # moe
            if ".mlp.experts.linear_fc" in mcore_weights_name:
                # get export index
                local_experts_idx = int(mcore_weights_name.split(".weight")[-1])
                num_experts = self.config.num_moe_experts
                num_experts_per_rank = num_experts // self.mpu.ep_size
                experts_idx = (
                    local_experts_idx + num_experts_per_rank * self.mpu.ep_rank
                )
                return hf_weights[0][experts_idx].clone().contiguous()
                # return hf_weights[0][experts_idx].T.clone().contiguous()
            elif "self_attention.out_norm.weight" in mcore_weights_name:
                return hf_weights[0] - 1

            return hf_weights[0]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.text_config.num_key_value_heads
            hidden_dim = self.hf_config.text_config.hidden_size
            num_attention_heads = self.hf_config.text_config.num_attention_heads
            if "vision_model" in mcore_weights_name:
                num_attention_heads = self.hf_config.text_config.vision_config.num_heads
                num_key_value_heads = self.hf_config.text_config.vision_config.num_heads
            head_dim = getattr(
                self.hf_config.text_config,
                "head_dim",
                hidden_dim // num_attention_heads,
            )
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            if self_attn_output_gate:
                real_num_key_value_heads = q.shape[0] // 2 // group_dim

                combined_w = q.reshape((num_attention_heads, 2 * head_dim, -1))
                q_w = combined_w.narrow(1, 0, head_dim).reshape((num_attention_heads * head_dim, -1))
                g_w = combined_w.narrow(1, head_dim, head_dim).reshape((num_attention_heads * head_dim, -1))

                q = q_w.view(
                    [
                        real_num_key_value_heads,
                        group_dim,
                        -1,
                    ]
                )
                g = g_w.view(
                    [
                        real_num_key_value_heads,
                        group_dim,
                        -1,
                    ]
                )
            else:
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

            if self_attn_output_gate:
                qkv = torch.cat([q, g, k, v], dim=1).view(*out_shape).contiguous()
            else:
                qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qkv
        elif (
            "vision_model" not in mcore_weights_name and (
                "linear_fc1.weight" in mcore_weights_name
                or "linear_fc1.bias" in mcore_weights_name
            )
        ):
            # merge gate_proj and up_proj
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        elif "self_attention.in_proj.weight" in mcore_weights_name:
            assert len(hf_weights) == 4
            hidden_size = self.hf_config.text_config.hidden_size
            in_proj_qkv, in_proj_z, in_proj_b, in_proj_a = hf_weights
            linear_num_key_heads = self.hf_config.text_config.linear_num_key_heads
            linear_key_head_dim = self.hf_config.text_config.linear_key_head_dim
            linear_num_value_heads = self.hf_config.text_config.linear_num_value_heads
            linear_value_head_dim = self.hf_config.text_config.linear_value_head_dim
            key_dim = linear_key_head_dim * linear_num_key_heads
            value_dim = linear_value_head_dim * linear_num_value_heads

            split_shape = [
                key_dim,
                key_dim,
                value_dim,
            ]
            wq, wk, wv = in_proj_qkv.split(split_shape, dim=0)

            # qkv_dim_per_partition = 2 * linear_key_head_dim + value_dim // linear_num_key_heads
            # in_proj_qkv_ = in_proj_qkv.reshape((linear_num_key_heads, qkv_dim_per_partition, -1))
            # wq = in_proj_qkv_.narrow(1, 0, linear_key_head_dim).reshape(key_dim, -1)
            # wk = in_proj_qkv_.narrow(1, linear_key_head_dim, linear_key_head_dim).reshape(key_dim, -1)
            # wv = in_proj_qkv_.narrow(1, 2 * linear_key_head_dim,
            #                          value_dim // linear_num_key_heads).reshape(value_dim, -1)


            wz = in_proj_z.reshape(value_dim, -1)
            wb = in_proj_b.reshape((linear_num_value_heads, -1))
            wa = in_proj_a.reshape((linear_num_value_heads, -1))
            return torch.cat([wq, wk, wv, wz, wb, wa], dim=0)
        else:
            print(f"others weights {len(hf_weights)} {[hf_w.shape for hf_w in hf_weights]}")
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
        if "mlp.experts.linear_fc" in mcore_weights_name:
            assert len(mcore_weights) == self.mpu.etp_size
            if self.mpu.etp_size == 1:
                assert len(mcore_weights) == 1
                return mcore_weights[0]
        else:
            assert len(mcore_weights) == self.mpu.tp_size
            if self.mpu.tp_size == 1:
                assert len(mcore_weights) == 1
                return mcore_weights[0]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            return torch.cat(mcore_weights, dim=0)
        elif (
            "vision_model" not in mcore_weights_name and (
                "linear_fc1.weight" in mcore_weights_name
                or "linear_fc1.bias" in mcore_weights_name
            )
        ):
            mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
            if not mcore_config.gated_linear_unit:
                return torch.cat(mcore_weights, dim=0)

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
        elif (
            "self_attention.linear_kv_down_proj.weight" in mcore_weights_name
            or "self_attention.linear_q_down_proj.weight" in mcore_weights_name
        ):
            # self_attention.linear_kv_down_proj.weight and self_attention.linear_q_down_proj.weight are copied
            return mcore_weights[0]
        elif "self_attention.in_proj.weight" in mcore_weights_name:
            mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
            tp_size = len(mcore_weights)
            k_dim = mcore_config.linear_num_key_heads * mcore_config.linear_key_head_dim
            v_dim = mcore_config.linear_num_value_heads * mcore_config.linear_value_head_dim
            split_shape = [
                k_dim // tp_size,
                k_dim // tp_size,
                v_dim // tp_size,
                v_dim // tp_size,
                mcore_config.linear_num_value_heads // tp_size,
                mcore_config.linear_num_value_heads // tp_size,
            ]
            # split_shape for [wq, wk, wv, wz, wb, wa]
            ret = self._split_weight_by_size_and_merge_across_tp(mcore_weights, split_shape)
        elif "self_attention.conv1d" in mcore_weights_name:
            if "weight" in mcore_weights_name:
                mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
                tp_size = len(mcore_weights)
                k_dim = mcore_config.linear_num_key_heads * mcore_config.linear_key_head_dim
                v_dim = mcore_config.linear_num_value_heads * mcore_config.linear_value_head_dim
                split_shape = [
                    k_dim // tp_size,
                    k_dim // tp_size,
                    v_dim // tp_size,
                ]
                # split_shape for [X, B, C]
                ret = self._split_weight_by_size_and_merge_across_tp(mcore_weights, split_shape)
            else:
                raise NotImplementedError(f"{mcore_weights_name} not supported yet")
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
        elif (
            "vision_model" not in mcore_weights_name and (
                "linear_fc1.weight" in mcore_weights_name
                or "linear_fc1.bias" in mcore_weights_name
            )
        ):
            mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
            if not mcore_config.gated_linear_unit:
                return mcore_weights.chunk(tp_split_size)

            gate, up = mcore_weights.chunk(2)
            gates = gate.chunk(tp_split_size)
            ups = up.chunk(tp_split_size)
            ret = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
        elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
            ret = mcore_weights.chunk(tp_split_size, dim=1)
        elif "self_attention.in_proj.weight" in mcore_weights_name:
            mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
            k_dim = mcore_config.linear_num_key_heads * mcore_config.linear_key_head_dim
            v_dim = mcore_config.linear_num_value_heads * mcore_config.linear_value_head_dim
            split_shape = [
                k_dim,
                k_dim,
                v_dim,
                v_dim,
                mcore_config.linear_num_value_heads,
                mcore_config.linear_num_value_heads,
            ]
            split_w_lst = mcore_weights.split(split_shape, dim=0)
            # split_w_lst: [wq, wk, wv, wz, wb, wa]
            assert len(split_w_lst) == 6, f"split_shape {split_shape} not supported"
            weight_list = []
            for weight in split_w_lst:
                weight_list.append(weight.chunk(tp_split_size))
            ret = [
                torch.cat([wq_slice, wk_slice, wv_slice, wz_slice, wb_slice, wa_slice], dim=0)
                for wq_slice, wk_slice, wv_slice, wz_slice, wb_slice, wa_slice in zip(*weight_list)
            ]
        elif "self_attention.conv1d" in mcore_weights_name:
            if "weight" in mcore_weights_name:
                mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
                k_dim = mcore_config.linear_num_key_heads * mcore_config.linear_key_head_dim
                v_dim = mcore_config.linear_num_value_heads * mcore_config.linear_value_head_dim
                split_shape = [
                    k_dim,
                    k_dim,
                    v_dim,
                ]
                split_w_lst = mcore_weights.split(split_shape, dim=0)
                # split_w_lst: [X, B, C]
                assert len(split_w_lst) == 3, f"split_shape {split_shape} not supported"
                weight_list = []
                for weight in split_w_lst:
                    weight_list.append(weight.chunk(tp_split_size))
                ret = [
                    torch.cat([x_slice, b_slice, c_slice], dim=0)
                    for x_slice, b_slice, c_slice in zip(*weight_list)
                ]
            else:
                raise NotImplementedError(f"{mcore_weights_name} not supported yet")
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

    def _split_weight_by_size_and_merge_across_tp(
        self,
        mcore_weights: list[torch.Tensor],
        split_shape: list[int],
    ) -> torch.Tensor:
        '''
        First split weight by splist_shape and then merge across tensor parallel ranks

        use for linear attn in_proj and linear attn conv1d layer weight
        '''
        tp_size = len(mcore_weights)

        weight_lst = [[] for _ in range(len(split_shape))]
        for mcore_weight in mcore_weights:
            split_w_lst = mcore_weight.split(split_shape, dim=0)
            assert len(split_w_lst) == len(weight_lst)
            for wi, split_w in enumerate(split_w_lst):
                weight_lst[wi].append(split_w)
        for weight in weight_lst:
            assert len(weight) == tp_size
        ret = torch.cat([torch.cat(w_split, dim=0) for w_split in weight_lst], dim=0)
        return ret

    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        """
        Creates and returns a model provider function.

        The returned function creates a GPTModel with the specified configuration
        when called with pre_process and post_process parameters.

        Args:
            post_model_creation_callbacks: List of callbacks to be called after model creation

        Returns:
            function: A provider function that creates and returns a GPTModel instance
        """

        share_embeddings_and_output_weights = getattr(
            self.hf_config, "tie_word_embeddings", False
        )

        def provider(
            pre_process,
            post_process,
            add_decoder=True,
            add_encoder=True,
            vp_stage: Optional[int] = None,
        ):
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)

            model = Qwen3p5VLModel(
                language_transformer_config=self.config,
                language_transformer_layer_spec=transformer_layer_spec,
                language_mtp_block_spec=None,
                language_vocab_size=self.hf_config.text_config.vocab_size,
                language_max_sequence_length=self.hf_config.text_config.max_position_embeddings,
                hf_config=self.hf_config,
                hf_vision_cls=self.HfVisionClass,
                language_rotary_base=self.hf_config.text_config.rope_scaling.get("rope_theta", 10000000),
                position_embedding_type="mrope",
                pre_process=pre_process,
                post_process=post_process,
                add_decoder=add_decoder,
                add_encoder=add_encoder,
                parallel_output=True,
                language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                image_token_id=self.hf_config.image_token_id,
                video_token_id=self.hf_config.video_token_id,
                vision_start_token_id=self.hf_config.vision_start_token_id,
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
