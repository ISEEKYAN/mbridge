from ..core import register_model
from .qwen3moe import Qwen3MoEBridge


@register_model("qwen3_next")
class Qwen3NextBridge(Qwen3MoEBridge):

    _ATTENTION_MAPPING = {
        **(Qwen3MoEBridge._ATTENTION_MAPPING),
        "self_attention.dt_bias": [
            "model.layers.{layer_number}.linear_attn.dt_bias"
        ],
        "self_attention.A_log": [
            "model.layers.{layer_number}.linear_attn.A_log"
        ],
        "self_attention.in_proj.weight": [
            "model.layers.{layer_number}.linear_attn.in_proj_qkvz.weight",
            "model.layers.{layer_number}.linear_attn.in_proj_ba.weight"
        ],
        "self_attention.conv1d.weight": [
            "model.layers.{layer_number}.linear_attn.conv1d.weight"
        ],
        "self_attention.out_norm.weight": [
            "model.layers.{layer_number}.linear_attn.norm.weight"
        ],
        "self_attention.out_proj.weight": [
            "model.layers.{layer_number}.linear_attn.out_proj.weight"
        ],
        "self_attention.in_proj.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"    
        ],
    }

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.

        Constructs a dictionary of arguments required to initialize a GPTModel
        based on the configuration.

        Returns:
            dict: A dictionary of arguments for GPTModel initialization
        """
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
            rotary_percent=self.hf_config.partial_rotary_factor,
        )

    def _build_config(self):
        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_experts,
            moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            moe_router_load_balancing_type="none",  # default None for RL
            moe_shared_expert_overlap=True,
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            moe_shared_expert_intermediate_size=self.hf_config.shared_expert_intermediate_size,
            moe_shared_expert_gate=self.hf_config.shared_expert_intermediate_size > 0,
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            layernorm_zero_centered_gamma=True,
            attention_output_gate=True,
            # Qwen3-next and linear attention
            kv_channels=self.hf_config.head_dim,
            linear_attention_type="gated_delta_net",
            linear_attention_freq=self.hf_config.full_attention_interval,
            linear_conv_kernel_dim=self.hf_config.linear_conv_kernel_dim,
            linear_key_head_dim=self.hf_config.linear_key_head_dim,
            linear_value_head_dim=self.hf_config.linear_value_head_dim,
            linear_num_key_heads=self.hf_config.linear_num_key_heads,
            linear_num_value_heads=self.hf_config.linear_num_value_heads,
            zero_centered_gated_delta_norm=False,
            #TODO: mtp 相关的参数还没加
        )
