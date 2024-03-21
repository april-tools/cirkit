# The implementations in
#     rg_node, region_graph, RandomBinaryTree, poon_domingos_structure
# are adapted from
#     https://github.com/cambridge-mlg/EinsumNetworks/blob/master/src/EinsumNetwork/Graph.py
# with extensive modifications.
# TODO: should this be here or separately? should we use comment or docstring?
#       also we need to update this at some time (already outdated)

# TODO: what else do we need to export?
from .region_graph import RegionGraph as RegionGraph
from .rg_node import PartitionNode as PartitionNode
from .rg_node import RegionNode as RegionNode
