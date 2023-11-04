from typing import List

import torch
from torch import Tensor

from cirkit.layers import Layer
from cirkit.region_graph import RegionNode


class ScopeLayer(Layer):
    """The scope layer.

    It re-orders unit activations such that they belong to the inputs of a circuit.
    """

    scope: Tensor  # To be registered as buffer

    def __init__(self, rg_nodes: List[RegionNode]):
        """Initialize a scope tensor.

        Args:
            rg_nodes: The list of region nodes on which it is defined.
        """
        super().__init__()
        self.num_vars = len(set(v for n in rg_nodes for v in n.scope))

        replica_indices = set(n.get_replica_idx() for n in rg_nodes)
        num_replicas = len(replica_indices)
        assert replica_indices == set(
            range(num_replicas)
        ), "Replica indices should be consecutive, starting with 0."

        scope = torch.zeros(self.num_vars, num_replicas, len(rg_nodes))
        for i, node in enumerate(rg_nodes):
            scope[list(node.scope), node.get_replica_idx(), i] = 1  # type: ignore[misc]
        self.register_buffer("scope", scope)

    def reset_parameters(self) -> None:
        """Do nothing.

        This layer does not have any parameters.
        """

    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Forward pass of the scope layer.

        Args:
            x (Tensor): The input units activations.

        Returns:
            Tensor: A folded tensor consisting of re-ordered unit activations.
        """
        # x: (batch_size, num_vars, num_units, num_replicas)
        # self.scope: (num_vars, num_replicas, num_folds)
        # output: (num_folds, num_units, batch_size)
        return torch.einsum("bdip,dpf->fib", x, self.scope)
