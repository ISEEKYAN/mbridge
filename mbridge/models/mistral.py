from ..core import register_model
from .llama import LLaMABridge


@register_model("mistral")
class MistralBridge(LLaMABridge):
    """
    Bridge implementation for dense Mistral models (Hugging Face model_type "mistral").

    Mistral checkpoints such as Mistral-Large-Instruct-2411 use the LLaMA layout: the same
    weight names, no qkv bias, and head_dim read from the config. Sliding-window attention
    has no counterpart here, so configs that use it are rejected rather than converted
    with full attention.
    """

    def _build_config(self):
        window = getattr(self.hf_config, "sliding_window", None)
        if window is not None and window < self.hf_config.max_position_embeddings:
            raise NotImplementedError(
                f"MistralBridge: sliding_window={window} is not supported "
                f"(max_position_embeddings={self.hf_config.max_position_embeddings})"
            )
        return super()._build_config()
