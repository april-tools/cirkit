import itertools
from collections import defaultdict
from typing import cast

from cirkit.templates.region_graph.algorithms.utils import HypercubeToScope
from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)
from cirkit.utils.scope import Scope


# pylint: disable-next=invalid-name
def QuadTree(shape: tuple[int, int, int], *, num_patch_splits: int = 2) -> RegionGraph:
    r"""Constructs a Quad Tree region graph.

    Args:
        shape: The image shape $(C, H, W)$, where $H$ is the height, $W$ is the width,
            and $C$ is the number of channels.
        num_patch_splits: The number of splits per patitioning, it can be either 2 or 4.

    Returns:
        RegionGraph: A Quad Tree region graph.

    Raises:
        ValueError: The image shape is not valid.
        ValueError: The number of patches to split is not valid.
    """
    return _QuadBuilder(shape, is_tree=True, num_patch_splits=num_patch_splits)


# pylint: disable-next=invalid-name
def QuadGraph(shape: tuple[int, int, int]) -> RegionGraph:
    r"""Constructs a Quad Graph region graph.

    Args:
        shape: The image shape $(C, H, W)$, where $H$ is the height, $W$ is the width,
            and $C$ is the number of channels.

    Returns:
        RegionGraph: A Quad Graph region graph.

    Raises:
        ValueError: The image shape is not valid.
    """
    return _QuadBuilder(shape, is_tree=False)


# pylint: disable-next=invalid-name
def _QuadBuilder(
    shape: tuple[int, int, int], *, is_tree: bool = False, num_patch_splits: int = 2
) -> RegionGraph:
    r"""Construct a RG with a quad tree.

    Args:
        shape: The image shape $(C, H, W)$, where $H$ is the height, $W$ is the width,
            and $C$ is the number of channels.
        is_tree: Whether the RG needs to be \
            structured-decomposable. Defaults to False.
        num_patch_splits: The number of patches to split. It can be either 2 or 4.
            This is used only when is_tree is True.

    Returns:
        RegionGraph: A region graph.

    Raises:
        ValueError: The image shape is not valid.
        ValueError: The number of patches to split is not valid.
    """
    if len(shape) != 3:
        raise ValueError("Quad Tree and Quad Graph region graphs only works for images")
    num_channels, height, width = shape
    if num_channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("The number of channels, the height and the width must be positive")
    if is_tree and num_patch_splits not in [2, 4]:
        raise ValueError("The number of patches to split must be either 2 or 4")

    # Padding using Scope({num_var}) which is one larger than range(num_var).
    # DISABLE: This is considered a constant here, although RegionNode is mutable.
    PADDING = RegionNode({height * width})  # pylint: disable=invalid-name
    # The regions of the current layer, in shape (H, W). The same PADDING object is reused.
    grid = [[PADDING] * (width + 1) for _ in range(height + 1)]

    # The list of region and partition nodes
    nodes: list[RegionGraphNode] = []

    # A map to each region/partition node to its children
    in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = defaultdict(list)

    # Add input region nodes
    # An object mapping rectangles of coordinates into variable scopes
    hypercube_to_scope = HypercubeToScope(shape)
    for i, j in itertools.product(range(height), range(width)):
        scope = hypercube_to_scope[((0, i, j), (num_channels, i + 1, j + 1))]
        rgn = RegionNode(scope)
        grid[i][j] = rgn
        nodes.append(rgn)

    def merge_regions_(rgn_in: list[RegionNode]) -> RegionNode:
        """Merge 2 or 4 regions to a larger region."""
        assert len(rgn_in) in {2, 4}
        scope = Scope.union(*tuple(rgn.scope for rgn in rgn_in))
        rgn = RegionNode(scope)
        ptn = PartitionNode(scope)
        nodes.append(rgn)
        nodes.append(ptn)
        in_nodes[rgn] = [ptn]
        in_nodes[ptn] = cast(list[RegionGraphNode], rgn_in)
        return rgn

    def merge_4_regions_tree_(rgn_in: list[RegionNode], *, num_patch_splits: int) -> RegionNode:
        # Merge 4 regions to a larger region, with structured-decomposablility
        assert len(rgn_in) == 4
        assert num_patch_splits in {2, 4}

        if num_patch_splits == 2:
            # Merge horizontally.
            region_top = merge_regions_(rgn_in[:2])
            region_bot = merge_regions_(rgn_in[2:])

            # Merge vertically.
            return merge_regions_([region_top, region_bot])

        # num_patch_splits == 4
        # Merge both horizontally and vertically
        return merge_regions_(rgn_in)

    def merge_4_regions_dag_(rgn_in: list[RegionNode]) -> RegionNode:
        # Merge 4 regions to a larger region, with non-structured-decomposable mix
        assert len(rgn_in) == 4

        # Merge horizontally, and then vertically.
        rgn_top = merge_regions_([rgn_in[0], rgn_in[1]])
        rgn_bot = merge_regions_([rgn_in[2], rgn_in[3]])
        rgn = merge_regions_([rgn_top, rgn_bot])  # Region node over the whole scope

        # Merge vertically, then horizontally, and reuse the same region node over the whole scope
        rgn_left = merge_regions_([rgn_in[0], rgn_in[2]])
        rgn_right = merge_regions_([rgn_in[1], rgn_in[3]])
        ptn = PartitionNode(rgn.scope)
        nodes.append(ptn)
        in_nodes[ptn] = [rgn_left, rgn_right]
        in_nodes[rgn].append(ptn)

        return rgn

    # Merge frontier by frontier, loop until (H, W)==(1, 1).
    while height > 1 or width > 1:
        height = (height + 1) // 2
        width = (width + 1) // 2
        prev_grid, grid = grid, [[PADDING] * (width + 1) for _ in range(height + 1)]

        for i, j in itertools.product(range(height), range(width)):
            regions = [  # Filter valid regions in the 4 possible sub-regions.
                rgn
                for rgn in (
                    prev_grid[i * 2][j * 2],
                    prev_grid[i * 2][j * 2 + 1],
                    prev_grid[i * 2 + 1][j * 2],
                    prev_grid[i * 2 + 1][j * 2 + 1],
                )
                if rgn != PADDING
            ]
            if len(regions) == 1:
                node = regions[0]
            elif len(regions) == 2:
                node = merge_regions_(regions)
            elif len(regions) == 4 and is_tree:
                node = merge_4_regions_tree_(regions, num_patch_splits=num_patch_splits)
            elif len(regions) == 4 and not is_tree:
                node = merge_4_regions_dag_(regions)
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "This should not happen"
            grid[i][j] = node

    return RegionGraph(nodes, in_nodes, outputs=[grid[0][0]])
