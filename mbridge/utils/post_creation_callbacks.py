from .layer import LinearForLastLayer


def make_value_model(model, pre_process, post_process, config, hf_config):
    if post_process:
        model.output_layer = LinearForLastLayer(
            input_size=config.hidden_size,
            output_size=1,
            config=config,
        )


def freeze_moe_router(model, pre_process, post_process, config, hf_config):
    for layer in model.decoder.layers:
        if hasattr(layer.mlp, "router"):
            router = layer.mlp.router
            for attr in ["weight", "bias", "expert_bias"]:
                param = getattr(router, attr, None)
                if param is not None:
                    param.requires_grad = False
        if hasattr(layer.mlp, "shared_experts"):
            shared_experts = layer.mlp.shared_experts
            for attr in ["gate_weight", "gate_bias"]:
                param = getattr(shared_experts, attr, None)
                if param is not None:
                    param.requires_grad = False
