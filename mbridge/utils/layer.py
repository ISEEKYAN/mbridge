# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch
from megatron.core import tensor_parallel


class LinearForLastLayer(torch.nn.Linear):
    """
    A custom linear layer implementation for the last layer of a model.

    This layer extends PyTorch's Linear module with functionality specifically designed
    for handling the final layer in transformer models with sequence parallelism.

    Attributes:
        sequence_parallel: Boolean indicating whether sequence parallelism is enabled
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        bias=False,
    ):
        """
        Initializes the LinearForLastLayer.

        Args:
            input_size: The size of the input features
            output_size: The size of the output features
            config: Configuration object containing parallelism settings
            bias: Whether to include a bias term (default: True)
        """
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

    def forward(
        self,
        input_,
        weight=None,
        runtime_gather_output=None,
    ):
        """
        Forward pass for the linear layer.

        This method computes the linear transformation and handles sequence parallelism
        if enabled, gathering outputs from different sequence parallel regions.

        Args:
            input_: Input tensor
            weight: Optional weight tensor to use instead of self.weight
            runtime_gather_output: Optional runtime parameter for gathering

        Returns:
            tuple: (logits, None) where logits is the output of the linear transformation
        """
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None
