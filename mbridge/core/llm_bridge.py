from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from .bridge import Bridge
from .layer import LinearForLastLayer


class LLMBridge(Bridge):
    """
    Bridge implementation for Large Language Models.

    This class extends the base Bridge class to provide specific functionality
    for handling Large Language Models (LLMs) like GPT models.
    """

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.

        Constructs a dictionary of arguments required to initialize a GPTModel
        based on the configuration.

        Returns:
            dict: A dictionary of arguments for GPTModel initialization
        """
        return dict(
            config=self.config,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )

    def _get_transformer_layer_spec(self):
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
        transformer_layer_spec = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True
        )
        return transformer_layer_spec

    def _model_provider(
        self, share_embeddings_and_output_weights=False, value_model=False
    ):
        """
        Creates and returns a model provider function.

        The returned function creates a GPTModel with the specified configuration
        when called with pre_process and post_process parameters.

        Args:
            share_embeddings_and_output_weights: Whether to share embedding weights
            value_model: Whether this is a value model with a custom output layer

        Returns:
            function: A provider function that creates and returns a GPTModel instance
        """

        def provider(pre_process, post_process):
            transformer_layer_spec = self._get_transformer_layer_spec()
            gptmodel_args = self._get_gptmodel_args()
            model = GPTModel(
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                **gptmodel_args,
            )
            if post_process and value_model:
                model.output_layer = LinearForLastLayer(
                    input_size=self.config.hidden_size,
                    output_size=1,
                    config=self.config,
                )

            return model

        return provider
