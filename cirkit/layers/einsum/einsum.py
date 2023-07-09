from abc import abstractmethod
from typing import Any, List

from torch import Tensor, nn

from cirkit.region_graph import PartitionNode

from ..layer import Layer

# TODO: relative import or absolute
# TODO: rework docstrings


class EinsumLayer(Layer):
    """Base for all einsums."""

    # TODO: kwargs should be public interface instead of `_`. How to supress this warning?
    #       all subclasses should accept all args as kwargs except for layer and k
    # TODO: subclasses should call reset_params -- where params are inited
    def __init__(  # type: ignore[misc]
        self,  # pylint: disable=unused-argument
        rg_nodes: List[PartitionNode],
        num_input_units: int,
        num_output_units: int,
        **kwargs: Any,
    ) -> None:
        """Init class.

        Args:
            rg_nodes (List[PartitionNode]): The region graph's partition node of the layer.
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            kwargs (Any): Passed to subclasses.
        """
        super().__init__()
        self.fold_count = len(rg_nodes)

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
    def forward(self, log_left: Tensor, log_right: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        Do EinsumLayer forward pass.

        We assume that all parameters are in the correct range (no checks done).

        Skeleton for each EinsumLayer (options Xa and Xb are mutual exclusive \
            and follows an a-path o b-path)
        1) Go To exp-space (with maximum subtraction) -> NON SPECIFIC
        2a) Do the einsum operation and go to the log space || 2b) Do the einsum operation
        3a) do the sum                                      || 3b) do the product
        4a) go to exp space do the einsum and back to log   || 4b) do the einsum operation [OPT]
        5a) do nothing                                      || 5b) back to log space

        :param log_left: value in log space for left child.
        :param log_right: value in log space for right child.
        :return: result of the left operations, in log-space.
        """
