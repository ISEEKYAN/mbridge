# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from transformers import AutoConfig

from .bridge import _MODEL_REGISTRY, BaseBridge


class AutoBridge(BaseBridge):
    """
    Automatically selects the appropriate model bridge class based on the model type.
    """

    @classmethod
    def from_pretrained(cls, hf_model_path):
        """
        Loads the appropriate bridge class from a pretrained model path.

        Args:
            hf_model_path: Hugging Face model path or identifier

        Returns:
            Bridge: An instance of the appropriate bridge class for the model
        """
        config = AutoConfig.from_pretrained(hf_model_path)

        return cls.from_config(config)

    @classmethod
    def from_config(cls, hf_config: AutoConfig):
        """
        Loads the appropriate bridge class from a Hugging Face configuration.

        Args:
            hf_config: Hugging Face model configuration

        Returns:
            Bridge: An instance of the appropriate bridge class for the model

        Raises:
            ValueError: If the model type is not registered in the model registry
        """
        model_type = hf_config.model_type
        if model_type in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[model_type](hf_config)
        else:
            raise ValueError(f"Unregistered model type: {model_type}")

    @classmethod
    def list_supported_models(cls):
        """
        Lists all supported model types.

        Returns:
            list: A list of supported model type strings
        """
        return list(_MODEL_REGISTRY.keys())

    def get_model(self, weight_path: str = None, **kwargs):
        """
        This is a placeholder implementation. AutoBridge doesn't implement this method directly
        as it delegates to specific bridge classes.

        Raises:
            NotImplementedError: Always raises this exception since AutoBridge is not meant
                to be instantiated directly.
        """
        raise NotImplementedError(
            "AutoBridge should not be instantiated directly. "
            "Use from_pretrained() or from_config() class methods instead."
        )

    def load_weights(self, models: list, weights_path: str) -> None:
        """
        This is a placeholder implementation. AutoBridge doesn't implement this method directly
        as it delegates to specific bridge classes.

        Raises:
            NotImplementedError: Always raises this exception since AutoBridge is not meant
                to be instantiated directly.
        """
        raise NotImplementedError(
            "AutoBridge should not be instantiated directly. "
            "Use from_pretrained() or from_config() class methods instead."
        )

    def set_extra_args(self, **kwargs):
        """
        This is a placeholder implementation. AutoBridge doesn't implement this method directly
        as it delegates to specific bridge classes.

        Raises:
            NotImplementedError: Always raises this exception since AutoBridge is not meant
                to be instantiated directly.
        """
        raise NotImplementedError(
            "AutoBridge should not be instantiated directly. "
            "Use from_pretrained() or from_config() class methods instead."
        )
