from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cirkit.region_graph import PartitionNode, RegionNode

from ..layer import Layer

# TODO: relative import or absolute
# TODO: rework docstrings


class _TwoInputs(NamedTuple):
    """Provide names for left and right inputs."""

    left: RegionNode
    right: RegionNode


class EinsumLayer(Layer):  # pylint: disable=too-many-instance-attributes
    """Base for all einsums."""

    # TODO: is product a good name here? should split
    # TODO: input can be more generic than List
    # TODO: subclasses should call reset_params -- where params are inited
    # we have to provide operation for input, operation for product and operation after product
    def __init__(  # type: ignore[misc]
        self, partition_layer: List[PartitionNode], k: int, **_: Any
    ) -> None:
        """Init class.

        Args:
            partition_layer (List[PartitionNode]): The current partition layer.
            k (int): The K.
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

        # get pairs of nodes which are input to the products (list of lists)
        # length of the outer list is same as self.products, length of inner lists is 2
        # "left child" has index 0, "right child" has index 1
        two_inputs = [_TwoInputs(*sorted(partition.inputs)) for partition in partition_layer]
        # TODO: again, why do we need sorting

        # collect all layers which contain left/right children
        self.left_addr = [inputs.left.einet_address for inputs in two_inputs]
        self.right_addr = [inputs.right.einet_address for inputs in two_inputs]

        # when the EinsumLayer is followed by a EinsumMixingLayer, we produce a
        # dummy "node" which outputs 0 (-inf in log-domain) for zero-padding.
        self.dummy_idx: Optional[int] = None

        # the dictionary mixing_component_idx stores which nodes (axis 2 of the
        # log-density tensor) need to get mixed
        # in the following EinsumMixingLayer
        self.mixing_component_idx: Dict[RegionNode, List[int]] = defaultdict(list)

        for c, product in enumerate(partition_layer):
            # each product must have exactly 1 parent (sum node)
            assert len(product.outputs) == 1
            out_region = product.outputs[0]

            if len(out_region.inputs) == 1:
                out_region.einet_address.layer = self
                out_region.einet_address.idx = c
            else:  # case followed by EinsumMixingLayer
                self.mixing_component_idx[out_region].append(c)
                self.dummy_idx = len(partition_layer)

        # TODO: correct way to init? definitely not in _forward()
        self.left_child_log_prob = torch.empty(())
        self.right_child_log_prob = torch.empty(())

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: U(0.01, 0.99)."""
        for param in self.parameters():
            nn.init.uniform_(param, 0.01, 0.99)

    @property
    @abstractmethod
    def clamp_value(self) -> float:
        """Value for parameters clamping to keep all probabilities greater than 0.

        :return: value for parameters clamping
        """

    @torch.no_grad()
    def clamp_params(self, clamp_all: bool = False) -> None:
        """Clamp parameters such that they are non-negative and \
        is impossible to get zero probabilities.

        This involves using a constant that is specific on the computation.

        Args:
            clamp_all (bool, optional): Whether to clamp all. Defaults to False.
        """
        for param in self.parameters():
            if clamp_all or param.requires_grad:
                param.clamp_(min=self.clamp_value)

    @property
    def params_shape(self) -> List[Tuple[int, ...]]:
        """Return all param shapes.

        Returns:
            List[Tuple[int, ...]]: All shapes.
        """
        return [param.shape for param in self.parameters()]

    @property
    def num_params(self) -> int:
        """Return the total number of parameters of the layer.

        :return: the total number of parameters of the layer.
        """
        return sum(param.numel() for param in self.parameters())

    @abstractmethod
    def _einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        """Compute the main Einsum operation of the layer.

        :param left_prob: value in log space for left child.
        :param right_prob: value in log space for right child.
        :return: result of the left operations, in log-space.
        """

    # TODO: input not used? also no return?
    def forward(self, _: Optional[Tensor] = None) -> None:
        """Do EinsumLayer forward pass.

        We assume that all parameters are in the correct range (no checks done).

        Skeleton for each EinsumLayer (options Xa and Xb are mutual exclusive \
            and follows an a-path o b-path)
        1) Go To exp-space (with maximum subtraction) -> NON SPECIFIC
        2a) Do the einsum operation and go to the log space || 2b) Do the einsum operation
        3a) do the sum                                      || 3b) do the product
        4a) go to exp space do the einsum and back to log   || 4b) do the einsum operation [OPT]
        5a) do nothing                                      || 5b) back to log space
        """
        # TODO: we should use dim=2, check all code
        self.left_child_log_prob = torch.stack(
            [addr.layer.prob[:, :, addr.idx] for addr in self.left_addr],
            dim=2,
        )
        self.right_child_log_prob = torch.stack(
            [addr.layer.prob[:, :, addr.idx] for addr in self.right_addr],
            dim=2,
        )

        # assert not torch.isinf(self.left_child_log_prob).any()
        # assert not torch.isinf(self.right_child_log_prob).any()
        # assert not torch.isnan(self.left_child_log_prob).any()
        # assert not torch.isnan(self.right_child_log_prob).any()

        # # # # # # # # # # STEP 1: Go To the exp space # # # # # # # # # #
        # We perform the LogEinsumExp trick, by first subtracting the maxes
        log_prob = self._einsum(self.left_child_log_prob, self.right_child_log_prob)

        # assert not torch.isinf(log_prob).any(), "Inf log prob"
        # assert not torch.isnan(log_prob).any(), "NaN log prob"

        # zero-padding (-inf in log-domain) for the following mixing layer
        if self.dummy_idx is not None:
            log_prob = F.pad(log_prob, [0, 1], "constant", float("-inf"))

        self.prob = log_prob
