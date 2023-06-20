from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

from torch import Tensor, nn
from torch.nn import functional as F

from cirkit.region_graph import PartitionNode, RegionNode

from ..layer import Layer

# TODO: relative import or absolute
# TODO: rework docstrings


class EinsumLayer(Layer):
    """Base for all einsums."""

    # TODO: is product a good name here? should split
    # TODO: kwargs should be public interface instead of `_`. How to supress this warning?
    #       all subclasses should accept all args as kwargs except for layer and k
    # TODO: subclasses should call reset_params -- where params are inited
    # we have to provide operation for input, operation for product and operation after product
    def __init__(  # type: ignore[misc]
        self,  # pylint: disable=unused-argument
        partition_layer: List[PartitionNode],
        k: int,
        **kwargs: Any,
    ) -> None:
        """Init class.

        Args:
            partition_layer (List[PartitionNode]): The current partition layer.
            k (int): The K.
            kwargs (Any): Passed to subclasses.
        """
        super().__init__()

        # # # # # # # # #
        #   CHECK
        # # # # # # # # #
        # TODO: do we really need this?

        # TODO: check all constructions that can use comprehension
        out_k = set(
            out_region.k for partition in partition_layer for out_region in partition.outputs
        )
        assert (
            len(out_k) == 1
        ), f"The K of output region nodes in the same layer must be the same, got {out_k}."

        # check if it is root  # TODO: what does this mean?
        if out_k.pop() > 1:
            self.out_k = k
            # set num_sums in the graph
            for partition in partition_layer:
                for out_region in partition.outputs:
                    out_region.k = k
        else:
            self.out_k = 1

        # TODO: why do we check it here?
        assert all(
            len(partition.inputs) == 2 for partition in partition_layer
        ), "Only 2-partitions are currently supported."

        in_k = set(in_region.k for partition in partition_layer for in_region in partition.inputs)
        assert (
            len(in_k) == 1
        ), f"The K of output region nodes in the same layer must be the same, got {in_k}."
        self.in_k = in_k.pop()

        # # # # # # # # #
        #   BUILD
        # # # # # # # # #

        # when the EinsumLayer is followed by a EinsumMixingLayer, we produce a
        # dummy "node" which outputs 0 (-inf in log-domain) for zero-padding.
        self.dummy_idx: Optional[int] = None

        # the dictionary mixing_component_idx stores which nodes (axis 2 of the
        # log-density tensor) need to get mixed
        # in the following EinsumMixingLayer
        self.mixing_component_idx: Dict[RegionNode, List[int]] = defaultdict(list)

        for part_idx, partition in enumerate(partition_layer):
            # each product must have exactly 1 parent (sum node)
            assert len(partition.outputs) == 1
            out_region = partition.outputs[0]

            if len(out_region.inputs) == 1:
                out_region.einet_address.layer = self
                out_region.einet_address.idx = part_idx
            else:  # case followed by EinsumMixingLayer
                self.mixing_component_idx[out_region].append(part_idx)
                self.dummy_idx = len(partition_layer)

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: U(0.01, 0.99)."""
        for param in self.parameters():
            nn.init.uniform_(param, 0.01, 0.99)

    @abstractmethod
    def _forward_einsum(self, log_left_prob: Tensor, log_right_prob: Tensor) -> Tensor:
        """Compute the main Einsum operation of the layer.

        :param log_left_prob: value in log space for left child.
        :param log_right_prob: value in log space for right child.
        :return: result of the left operations, in log-space.
        """

    # TODO: find a better way to do this override
    # pylint: disable=arguments-differ
    def forward(  # type: ignore[override]
        self, log_left_prob: Tensor, log_right_prob: Tensor
    ) -> Tensor:
        """Do EinsumLayer forward pass.

        We assume that all parameters are in the correct range (no checks done).

        Skeleton for each EinsumLayer (options Xa and Xb are mutual exclusive \
            and follows an a-path o b-path)
        1) Go To exp-space (with maximum subtraction) -> NON SPECIFIC
        2a) Do the einsum operation and go to the log space || 2b) Do the einsum operation
        3a) do the sum                                      || 3b) do the product
        4a) go to exp space do the einsum and back to log   || 4b) do the einsum operation [OPT]
        5a) do nothing                                      || 5b) back to log space

        :param log_left_prob: value in log space for left child.
        :param log_right_prob: value in log space for right child.
        :return: result of the left operations, in log-space.
        """
        log_prob = self._forward_einsum(log_left_prob, log_right_prob)

        if self.dummy_idx is not None:
            log_prob = F.pad(log_prob, [0, 1], "constant", float("-inf"))

        self.prob = log_prob
        return self.prob
