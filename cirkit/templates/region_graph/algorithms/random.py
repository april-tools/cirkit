import itertools
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)
from cirkit.utils.scope import Scope


# pylint: disable-next=invalid-name
def RandomBinaryTree(
    num_variables: int, *, depth: int | None = None, num_repetitions: int = 1, seed: int = 42
) -> RegionGraph:
    """Construct a RG with random binary trees.

    See:
        - *Random sum-product networks: A simple but effective approach to probabilistic deep learning.*  [🔗](https://proceedings.mlr.press/v115/peharz20a.html)  
          Robert Peharz, Antonio Vergari, Karl Stelzner, Alejandro Molina, Xiaoting Shao, Martin Trapp, Kristian Kersting, and Zoubin Ghahramani.  
          In Uncertainty in Artificial Intelligence, pp. 334-344. PMLR, 2020.

    Args:
        num_variables (int): The number of variables in the RG.
        depth (int): The depth of the binary tree. If None, the maximum possible depth is used.
        num_repetitions (int): The number of repetitions of binary trees (degree of root).
        seed: The seed to initialize the random state.

    Returns:
        RegionGraph: A randomized binary tree region graph.

    Raises:
        ValueError: If either the number of variables or number of reptitions are not positive.
        ValueError: If the given depth is either negative or greate than the maximum allowed one.
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
    nodes: list[RegionGraphNode] = [root]
    in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = defaultdict(list)

    def random_scope_partitioning(
        scope: Scope,
        num_parts: int | None = None,
        proportions: Sequence[float] | None = None,
    ) -> list[Scope]:
        # Shuffle the region node scope
        scope_ls = list(scope)
        random_state.shuffle(scope_ls)

        # ANNOTATE: Numpy has typing issues.
        split: np.ndarray  # Unnormalized split points including 0 and 1.
        if proportions is None:
            if num_parts is None:
                raise ValueError("Must provide at least one of num_parts and proportions")
            split = np.arange(num_parts + 1, dtype=np.float64)
        else:
            split = np.array([0.0] + list(proportions), dtype=np.float64).cumsum()

        # ANNOTATE: ndarray.tolist gives Any.
        # CAST: Numpy has typing issues.
        # IGNORE: Numpy has typing issues.
        split_point: list[int] = (
            (split / split[-1] * len(scope_ls)).round().astype(np.int64).tolist()
        )

        # ANNOTATE: Specify content for empty container.
        scopes: list[Scope] = []
        for l, r in itertools.pairwise(split_point):
            if l < r:  # A region must have as least one var, otherwise we skip it.
                scopes.append(Scope(scope_ls[l:r]))

        if len(scopes) == 1:
            # Only one region, meaning cannot partition anymore, and we just keep the original
            # node as the leaf.
            return [Scope(scope_ls)]

        return scopes

    for _ in range(num_repetitions):
        frontier: list[RegionGraphNode] = [root]
        for _ in range(depth):
            next_frontier: list[RegionGraphNode] = []
            for rgn in frontier:
                partition_node = PartitionNode(rgn.scope)
                scopes = random_scope_partitioning(rgn.scope, num_parts=2)
                if len(scopes) == 1:  # No further binary partitionings are possible
                    continue
                region_nodes: list[RegionGraphNode] = [RegionNode(scope) for scope in scopes]
                nodes.append(partition_node)
                nodes.extend(region_nodes)
                in_nodes[rgn].append(partition_node)
                in_nodes[partition_node] = region_nodes
                next_frontier.extend(region_nodes)
            frontier = next_frontier

    return RegionGraph(nodes, in_nodes, outputs=[root])
