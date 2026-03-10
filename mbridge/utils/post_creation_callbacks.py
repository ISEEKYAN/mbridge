from .layer import LinearForLastLayer

def unwrap_language_model(model):
    if hasattr(model, "language_model"):
        return model.language_model
    return model

def make_value_model(model, pre_process, post_process, config, hf_config):
    model = unwrap_language_model(model)
    if post_process:
        model.output_layer = LinearForLastLayer(
            input_size=config.hidden_size,
            output_size=1,
            config=config,
        )


def freeze_moe_router(model, pre_process, post_process, config, hf_config):
    model = unwrap_language_model(model)
    for layer in model.decoder.layers:
        if hasattr(layer.mlp, "router"):
            if hasattr(layer.mlp.router, "weight"):
                layer.mlp.router.weight.requires_grad = False
            if hasattr(layer.mlp.router, "bias") and layer.mlp.router.bias is not None:
                layer.mlp.router.bias.requires_grad = False
        if hasattr(layer.mlp, "shared_experts"):
            if hasattr(layer.mlp.shared_experts, "gate_weight"):
                layer.mlp.shared_experts.gate_weight.requires_grad = False
            if hasattr(layer.mlp.shared_experts, "gate_bias"):
                layer.mlp.shared_experts.gate_bias.requires_grad = False
