import itertools
from typing import Tuple

from cirkit.templates.region_graph import RegionNode
from cirkit.templates.region_graph.graph import RegionGraph
from cirkit.templates.region_graph.utils import HypercubeToScope

# TODO: now should work with H!=W but need tests


def _merge_2_regions(graph: RegionGraph, region_nodes: Tuple[RegionNode, RegionNode]) -> RegionNode:
    """Merge 2 regions to a larger region.

    Args:
        graph (RegionGraph): The region graph to hold the merging.
        region_nodes (Tuple[cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode]): The region nodes to merge.

    Returns:
        cirkit.templates.region_graph.RegionNode: The merged region node.
    """
    assert len(region_nodes) == 2

    region_node = RegionNode(region_nodes[0].scope | region_nodes[1].scope)
    graph.add_partitioning(region_node, region_nodes)
    return region_node


def _merge_4_regions(
    graph: RegionGraph, region_nodes: Tuple[RegionNode, RegionNode, RegionNode, RegionNode]
) -> RegionNode:
    """Merge 4 regions to a larger region.

    Args:
        graph (RegionGraph): The region graph to hold the merging.
        region_nodes (Tuple[cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode]): The region nodes to merge.

    Returns:
        cirkit.templates.region_graph.RegionNode: The merged region node.
    """
    assert len(region_nodes) == 4

    region_node = RegionNode(
        region_nodes[0].scope
        | region_nodes[1].scope
        | region_nodes[2].scope
        | region_nodes[3].scope
    )
    graph.add_partitioning(region_node, region_nodes)
    return region_node


def _merge_4_regions_tree(
    graph: RegionGraph,
    region_nodes: Tuple[RegionNode, RegionNode, RegionNode, RegionNode],
    num_patch_splits: int = 2,
) -> RegionNode:
    """Merge 4 regions to a larger region, with structured-decomposablility.

    We first merge horizontally and then vertically.

    Args:
        graph (RegionGraph): The region graph to hold the merging.
        region_nodes (Tuple[cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode]): The region nodes to \
            merge.
        num_patch_splits (int): The number of patches to split. It can be either 2 or 4.

    Returns:
        cirkit.templates.region_graph.RegionNode: The merged region node.
    """
    assert len(region_nodes) == 4
    assert num_patch_splits in [2, 4], "The number of patches to split must be either 2 or 4"

    if num_patch_splits == 2:
        # Merge horizontally.
        region_top = _merge_2_regions(graph, region_nodes[:2])
        region_bot = _merge_2_regions(graph, region_nodes[2:])

        # Merge vertically.
        region_whole = _merge_2_regions(graph, (region_top, region_bot))
    else:  # num_patch_splits == 4
        # Merge both horizontally and vertically
        region_whole = _merge_4_regions(graph, region_nodes)

    return region_whole


def _merge_4_regions_mixed(
    graph: RegionGraph, region_nodes: Tuple[RegionNode, RegionNode, RegionNode, RegionNode]
) -> RegionNode:
    """Merge 4 regions to a larger region, with non-structured-decomposable mixutre.

    Args:
        graph (RegionGraph): The region graph to hold the merging.
        region_nodes (Tuple[cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode, cirkit.templates.region_graph.RegionNode]): The region nodes to \
            merge.

    Returns:
        cirkit.templates.region_graph.RegionNode: The merged region node.
    """
    assert len(region_nodes) == 4

    # Merge horizontally then vertically.
    region_whole = _merge_4_regions_tree(graph, region_nodes)

    # Merge vertically then horizontally.
    region_lft = _merge_2_regions(graph, region_nodes[0::2])
    region_rit = _merge_2_regions(graph, region_nodes[1::2])
    # Reuse the region_whole that is already constructed.
    graph.add_partitioning(region_whole, (region_lft, region_rit))

    return region_whole


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def QuadGraph(
    shape: Tuple[int, int], *, is_tree: bool = False, num_patch_splits: int = 2
) -> RegionGraph:
    """Construct a RG with a quad tree.

    Args:
        shape (Tuple[int, int]): The shape of the image, in (H, W).
        is_tree (bool, optional): Whether the RG needs to be \
            structured-decomposable. Defaults to False.
        num_patch_splits (int): The number of patches to split. It can be either 2 or 4.
            This is used only when is_tree is True.

    Returns:
        RegionGraph: The QT RG.
    """
    assert len(shape) == 2, "QT only works for 2D image."
    height, width = shape

    assert height > 0 and width > 0, "The image must have positive size."

    graph = RegionGraph()
    hypercube_to_scope = HypercubeToScope(shape)

    # Padding using Scope({num_var}) which is one larger than range(num_var).
    # DISABLE: This is considered a constant here, although RegionNode is mutable.
    PADDING = RegionNode({height * width})  # pylint: disable=invalid-name
    # The regions of the current layer, in shape (H, W). The same PADDING object is reused.
    layer = [[PADDING] * (width + 1) for _ in range(height + 1)]

    # Add univariate input nodes.
    for i, j in itertools.product(range(height), range(width)):
        node = RegionNode(hypercube_to_scope[((i, j), (i + 1, j + 1))])
        layer[i][j] = node
        graph.add_node(node)

    # Merge layer by layer, loop until (H, W)==(1, 1).
    for _ in range((max(height, width) - 1).bit_length()):
        height = (height + 1) // 2
        width = (width + 1) // 2
        prev_layer, layer = layer, [[PADDING] * (width + 1) for _ in range(height + 1)]

        for i, j in itertools.product(range(height), range(width)):
            regions = tuple(  # Filter valid regions in the 4 possible sub-regions.
                node
                for node in (
                    prev_layer[i * 2][j * 2],
                    prev_layer[i * 2][j * 2 + 1],
                    prev_layer[i * 2 + 1][j * 2],
                    prev_layer[i * 2 + 1][j * 2 + 1],
                )
                if node != PADDING
            )
            if len(regions) == 1:
                node = regions[0]
            elif len(regions) == 2:
                node = _merge_2_regions(graph, regions)
            elif len(regions) == 4 and is_tree:
                node = _merge_4_regions_tree(graph, regions, num_patch_splits=num_patch_splits)
            elif len(regions) == 4 and not is_tree:
                node = _merge_4_regions_mixed(graph, regions)
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "This should not happen."
            layer[i][j] = node

    return graph.freeze()
