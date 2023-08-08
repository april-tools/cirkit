from abc import abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor, nn

from cirkit.layers.layer import Layer

# TODO: relative import or absolute
# TODO: rework docstrings


class SumProductLayer(Layer):
    """Base for all "fused" sum-product layers."""

    # TODO: kwargs should be public interface instead of `_`. How to supress this warning?
    #       all subclasses should accept all args as kwargs except for layer and k
    # TODO: subclasses should call reset_params -- where params are inited
    def __init__(  # type: ignore[misc]
        self,  # pylint: disable=unused-argument
        num_input_units: int,
        num_output_units: int,
        num_folds: int = 1,
        fold_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            num_folds (int): The number of folds.
            fold_mask (Optional[torch.Tensor]): The mask to apply to the folded parameter tensors.
            kwargs (Any): Passed to subclasses.
        """
        super().__init__(num_folds=num_folds, fold_mask=fold_mask)
        assert num_input_units > 0
        assert num_output_units > 0
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: U(0.01, 0.99)."""
        for param in self.parameters():
            nn.init.uniform_(param, 0.01, 0.99)

    # TODO: find a better way to do this override
    # TODO: what about abstract?
    @abstractmethod
    # pylint: disable-next=arguments-differ
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main einsum operation of the layer.

        Do SumProductLayer forward pass.

        We assume that all parameters are in the correct range (no checks done).

        Skeleton for each SumProductLayer (options Xa and Xb are mutual exclusive \
            and follows an a-path o b-path)
        1) Go To exp-space (with maximum subtraction) -> NON SPECIFIC
        2a) Do the einsum operation and go to the log space || 2b) Do the einsum operation
        3a) do the sum                                      || 3b) do the product
        4a) go to exp space do the einsum and back to log   || 4b) do the einsum operation [OPT]
        5a) do nothing                                      || 5b) back to log space

        :param inputs: the input tensor.
        :return: result of the left operations, in log-space.
        """
