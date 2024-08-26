from typing import Dict, List, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)
from cirkit.utils.scope import Scope


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def FullyFactorized(num_variables: int, *, num_repetitions: int = 1) -> RegionGraph:
    """Construct a region graph with fully factorized partitions.

    Args:
        num_variables: The number of variables in the RG.
        num_repetitions: The number of fully factorized partitions. Defaults to 1.

    Returns:
        RegionGraph: The fully-factorized region graph.
    """
    if num_variables <= 0:
        raise ValueError("The number of variables must be positive")
    if num_repetitions <= 0:
        raise ValueError("The number of repetitions must be positive")
    root = RegionNode(range(num_variables))
    nodes = [root]
    in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = {root: []}
    if num_variables > 1:
        for _ in range(num_repetitions):
            partition_node = PartitionNode(range(num_variables))
            leaf_nodes = [RegionNode([vid]) for vid in range(num_variables)]
            in_nodes[partition_node] = leaf_nodes
            in_nodes[root].append(partition_node)
            nodes.extend(leaf_nodes)
            nodes.append(partition_node)

    return RegionGraph(nodes, in_nodes, [root])


def RandomBinaryTree(
    num_variables: int, *, depth: Optional[int] = None, num_repetitions: int = 1, seed: int = 42
) -> RegionGraph:
    """Construct a RG with random binary trees.

    See:
        Random sum-product networks: A simple but effective approach to probabilistic deep learning.
        Robert Peharz, Antonio Vergari, Karl Stelzner, Alejandro Molina, Xiaoting Shao,
        Martin Trapp, Kristian Kersting, Zoubin Ghahramani.
        UAI 2019.

    Args:
        num_variables (int): The number of variables in the RG.
        depth (int): The depth of the binary tree. If None, the maximum possible depth is used.
        num_repetitions (int): The number of repetitions of binary trees (degree of root).

    Returns:
        RegionGraph: The RBT RG.
    """
    if num_variables <= 0:
        raise ValueError("The number of variables must be positive")
    if num_repetitions <= 0:
        raise ValueError("The number of repetitions must be positive")
    max_depth = int(np.ceil(np.log2(num_variables)))
    if depth is None:
        depth = max_depth
    elif depth < 0 or depth > max_depth:
        raise ValueError(f"The depth must be between 0 and {max_depth}")
    random_state = np.random.RandomState(seed)
    root = RegionNode(range(num_variables))
    nodes = [root]
    in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = {root: []}

    def random_scope_partitioning(
        scope: Scope,
        num_parts: Optional[int] = None,
        proportions: Optional[Sequence[float]] = None,
    ) -> List[Scope]:
        # Shuffle the region node scope
        scope = list(scope)
        random_state.shuffle(scope)

        # ANNOTATE: Numpy has typing issues.
        split: NDArray[np.float64]  # Unnormalized split points including 0 and 1.
        if proportions is None:
            if num_parts is None:
                raise ValueError("Must provide at least one of num_parts and proportions")
            split = np.arange(num_parts + 1, dtype=np.float64)
        else:
            split = np.array([0.0] + list(proportions), dtype=np.float64).cumsum()

        # ANNOTATE: ndarray.tolist gives Any.
        # CAST: Numpy has typing issues.
        # IGNORE: Numpy has typing issues.
        split_point: List[int] = (
            cast(NDArray[np.float64], split / split[-1] * len(scope))  # type: ignore[misc]
            .round()
            .astype(np.int64)
            .tolist()
        )

        # ANNOTATE: Specify content for empty container.
        scopes: List[Scope] = []
        for l, r in zip(split_point[:-1], split_point[1:]):
            # FUTURE: for l, r in itertools.pairwise(split_point) in 3.10
            if l < r:  # A region must have as least one var, otherwise we skip it.
                scopes.append(Scope(scope[l:r]))

        if len(scopes) == 1:
            # Only one region, meaning cannot partition anymore, and we just keep the original node as
            # the leaf.
            return [Scope(scope)]

        return scopes

    for _ in range(num_repetitions):
        frontier = [root]
        for _ in range(depth):
            next_frontier = []
            for rgn in frontier:
                partition_node = PartitionNode(rgn.scope)
                scopes = random_scope_partitioning(rgn.scope, num_parts=2)
                if len(scopes) == 1:  # No further binary partitionings are possible
                    continue
                region_nodes = [RegionNode(scope) for scope in scopes]
                nodes.append(partition_node)
                nodes.extend(region_nodes)
                if rgn == root:
                    in_nodes[rgn].append(partition_node)
                else:
                    in_nodes[rgn] = [partition_node]
                in_nodes[partition_node] = region_nodes
                next_frontier.extend(region_nodes)
            frontier = next_frontier

    return RegionGraph(nodes, in_nodes, outputs=[root])
