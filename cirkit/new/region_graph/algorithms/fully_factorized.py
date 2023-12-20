from cirkit.new.region_graph.region_graph import RegionGraph
from cirkit.new.region_graph.rg_node import RegionNode


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def FullyFactorized(*, num_vars: int, num_repetitions: int = 1) -> RegionGraph:
    """Construct a RG with fully factorized partitions.

    Args:
        num_vars (int): The number of variables in the RG.
        num_repetitions (int, optional): The number of fully factorized partitions. Defaults to 1.

    Returns:
        RegionGraph: The FF RG.
    """
    graph = RegionGraph()
    root = RegionNode(range(num_vars))
    graph.add_node(root)

    if num_vars > 1:  # For num_vars==1, no partition is needed.
        for _ in range(num_repetitions):
            graph.add_partitioning(root, (RegionNode((var,)) for var in range(num_vars)))

    return graph.freeze()
