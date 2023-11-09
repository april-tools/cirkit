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
    """The scope ordering, shape (D, P, F)."""

    def __init__(self, rg_nodes: List[RegionNode]):
        """Initialize a scope tensor.

        Args:
            rg_nodes: The list of region nodes on which it is defined.
        """
        # TODO: what should be here? none of them is used so current all 1s
        super().__init__(num_input_units=1, num_output_units=1, arity=1, num_folds=1)
        self.num_vars = len(set(v for n in rg_nodes for v in n.scope))

        replica_indices = set(n.replica_idx for n in rg_nodes)
        num_replicas = len(replica_indices)
        assert replica_indices == set(
            range(num_replicas)
        ), "Replica indices should be consecutive, starting with 0."

        scope = torch.zeros(self.num_vars, num_replicas, len(rg_nodes))
        for i, node in enumerate(rg_nodes):
            scope[list(node.scope), node.replica_idx, i] = 1  # type: ignore[misc]
        self.register_buffer("scope", scope)

    def reset_parameters(self) -> None:
        """Do nothing.

        This layer does not have any parameters.
        """

    # TODO: any good way to sync the docstring?
    # TODO: override is due to integral layer
    def __call__(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Invoke the forward function.

        Args:
            x (Tensor): The input to this layer, shape (B, D, K, P).

        Returns:
            Tensor: The output of this layer, shape (F, K, B).
        """
        return super().__call__(x)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (B, D, K, P).

        Returns:
            Tensor: The output of this layer, shape (F, K, B).
        """
        return torch.einsum("bdkp,dpf->fkb", x, self.scope)
