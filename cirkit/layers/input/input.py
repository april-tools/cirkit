from typing import Any, List

from cirkit.layers import Layer
from cirkit.region_graph import RegionNode


class InputLayer(Layer):
    """Abstract input layer class."""

    def __init__(self, rg_nodes: List[RegionNode], **_: Any):  # type: ignore[misc]
        """Initialize an input layer.

        Args:
            rg_nodes: The region nodes on which it is defined.
            **kwargs: Unused.
        """
        super().__init__()
        self.rg_nodes = rg_nodes
        self.num_vars = len(set(v for n in rg_nodes for v in n.scope))
