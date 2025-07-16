from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def FullyFactorized(num_variables: int, *, num_repetitions: int = 1) -> RegionGraph:
    """Construct a region graph with fully factorized partitions.

    Args:
        num_variables: The number of variables in the RG.
        num_repetitions: The number of fully factorized partitions. Defaults to 1.

    Returns:
        RegionGraph: The fully-factorized region graph.

    Raises:
        ValueError: If either the number of variables or number of reptitions are not positive.
    """
    if num_variables <= 0:
        raise ValueError("The number of variables must be positive")
    if num_repetitions <= 0:
        raise ValueError("The number of repetitions must be positive")

    root = RegionNode(range(num_variables))
    nodes: list[RegionGraphNode] = [root]
    in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = {root: []}
    if num_variables == 1:
        return RegionGraph(nodes, in_nodes, [root])

    for _ in range(num_repetitions):
        partition_node = PartitionNode(range(num_variables))
        leaf_nodes: list[RegionGraphNode] = [RegionNode([vid]) for vid in range(num_variables)]
        in_nodes[partition_node] = leaf_nodes
        in_nodes[root].append(partition_node)
        nodes.extend(leaf_nodes)
        nodes.append(partition_node)

    return RegionGraph(nodes, in_nodes, [root])
