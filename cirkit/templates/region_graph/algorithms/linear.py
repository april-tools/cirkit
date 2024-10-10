from collections import defaultdict

import numpy as np

from cirkit.templates.region_graph.graph import (
    RegionGraph,
    RegionNode,
    RegionGraphNode,
    PartitionNode,
)
from cirkit.utils.scope import Scope


# pylint: disable-next=invalid-name
def LinearTree(
    num_variables: int,
    *,
    num_repetitions: int = 1,
    ordering: list[int] = None,
    randomize: bool = False,
    seed: int = 42,
) -> RegionGraph:
    """Construct a linear tree region graph, where each partitioning conditions on a single
     variable at a time.

    Args:
        num_variables: The number of variables in the RG.
        num_repetitions: The number of repeated linear trees. Defaults to 1.
        ordering: The ordering of variables. If it is None, then it is assumed to be the natural
            ordering.
        randomize: Whether to randomize the variable ordering for each repetition.
        seed: The seed to use in case of randomize being True.

    Returns:
        RegionGraph: The linear tree region graph.

    Raises:
        ValueError: If either the number of variables or number of reptitions are not positive.
        ValueError: If the given variable ordering is not valid.
    """
    if num_variables <= 0:
        raise ValueError("The number of variables must be positive")
    if num_repetitions <= 0:
        raise ValueError("The number of repetitions must be positive")
    if ordering is not None and sorted(ordering) != list(range(num_variables)):
        raise ValueError(
            f"The variables ordering must be a permutation of values from 0 to {num_variables-1}"
        )

    root = RegionNode(range(num_variables))
    nodes: list[RegionGraphNode] = [root]
    in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = defaultdict(list)
    if num_variables == 1:
        return RegionGraph(nodes, in_nodes, [root])

    if ordering is None:
        ordering = list(range(num_variables))
    random_state = np.random.RandomState(seed) if randomize else None
    for _ in range(num_repetitions):
        if randomize:
            assert random_state is not None
            random_state.shuffle(ordering)
        node = root
        for vid in ordering[:-1]:
            scope = list(node.scope)
            partition_node = PartitionNode(Scope(scope))
            scope.remove(vid)
            leaf_node = RegionNode(Scope([vid]))
            next_node = RegionNode(Scope(scope))
            nodes.append(partition_node)
            nodes.append(leaf_node)
            nodes.append(next_node)
            in_nodes[node].append(partition_node)
            in_nodes[partition_node] = [leaf_node, next_node]
            node = next_node

    return RegionGraph(nodes, in_nodes, [root])
