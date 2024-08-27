import itertools
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)
from cirkit.templates.region_graph.utils import HyperCube, HypercubeToScope
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
    nodes: List[RegionGraphNode] = [root]
    in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = {root: []}
    if num_variables == 1:
        return RegionGraph(nodes, in_nodes, [root])

    for _ in range(num_repetitions):
        partition_node = PartitionNode(range(num_variables))
        leaf_nodes = [RegionNode([vid]) for vid in range(num_variables)]
        in_nodes[partition_node] = leaf_nodes
        in_nodes[root].append(partition_node)
        nodes.extend(leaf_nodes)
        nodes.append(partition_node)

    return RegionGraph(nodes, in_nodes, [root])


def LinearTree(
    num_variables: int,
    *,
    num_repetitions: int = 1,
    ordering: List[int] = None,
    randomize: bool = False,
    seed: int = 42,
) -> RegionGraph:
    if num_variables <= 0:
        raise ValueError("The number of variables must be positive")
    if num_repetitions <= 0:
        raise ValueError("The number of repetitions must be positive")
    if ordering is not None and sorted(ordering) != list(range(num_variables)):
        raise ValueError(
            f"The variables ordering must be a permutation of values from 0 to {num_variables-1}"
        )

    root = RegionNode(range(num_variables))
    nodes: List[RegionGraphNode] = [root]
    in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = defaultdict(list)
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
    in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = defaultdict(list)

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
                in_nodes[rgn].append(partition_node)
                in_nodes[partition_node] = region_nodes
                next_frontier.extend(region_nodes)
            frontier = next_frontier

    return RegionGraph(nodes, in_nodes, outputs=[root])


def QuadTree(shape: Tuple[int, int], *, num_patch_splits: int = 2) -> RegionGraph:
    return _QuadBuilder(shape, is_tree=True, num_patch_splits=num_patch_splits)


def QuadGraph(shape: Tuple[int, int]) -> RegionGraph:
    return _QuadBuilder(shape, is_tree=False)


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def _QuadBuilder(
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
    if len(shape) != 2:
        raise ValueError("QT only works for 2D image.")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers")
    if is_tree and num_patch_splits not in [2, 4]:
        raise ValueError("The number of patches to split must be either 2 or 4")

    # An object mapping rectangles of coordinates into variable scopes
    hypercube_to_scope = HypercubeToScope(shape)

    # Padding using Scope({num_var}) which is one larger than range(num_var).
    # DISABLE: This is considered a constant here, although RegionNode is mutable.
    PADDING = RegionNode({height * width})  # pylint: disable=invalid-name
    # The regions of the current layer, in shape (H, W). The same PADDING object is reused.
    grid = [[PADDING] * (width + 1) for _ in range(height + 1)]

    # The list of region and partition nodes
    nodes: List[RegionGraphNode] = []

    # A map to each region/partition node to its children
    in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = defaultdict(list)

    # Add univariate input region nodes
    for i, j in itertools.product(range(height), range(width)):
        rgn = RegionNode(hypercube_to_scope[((i, j), (i + 1, j + 1))])
        grid[i][j] = rgn
        nodes.append(rgn)

    def merge_regions_(rgn_in: List[RegionNode]) -> RegionNode:
        """Merge 2 or 4 regions to a larger region."""
        assert len(rgn_in) in [2, 4]
        scope = Scope.union(*tuple(rgn.scope for rgn in rgn_in))
        rgn = RegionNode(scope)
        ptn = PartitionNode(scope)
        nodes.append(rgn)
        nodes.append(ptn)
        in_nodes[rgn] = [ptn]
        in_nodes[ptn] = rgn_in
        return rgn

    def merge_4_regions_tree_(rgn_in: List[RegionNode], *, num_patch_splits: int) -> RegionNode:
        """Merge 4 regions to a larger region, with structured-decomposablility."""
        assert len(rgn_in) == 4
        assert num_patch_splits in [2, 4]

        if num_patch_splits == 2:
            # Merge horizontally.
            region_top = merge_regions_(rgn_in[:2])
            region_bot = merge_regions_(rgn_in[2:])

            # Merge vertically.
            return merge_regions_([region_top, region_bot])

        # num_patch_splits == 4
        # Merge both horizontally and vertically
        return merge_regions_(rgn_in)

    def merge_4_regions_dag_(rgn_in: List[RegionNode]) -> RegionNode:
        """Merge 4 regions to a larger region, with non-structured-decomposable mixutre."""
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


def _parse_poon_domingos_delta(
    delta: Union[float, List[float], List[List[float]]], shape: Sequence[int], axes: Sequence[int]
) -> List[List[List[int]]]:
    """Parse the delta argument into cut points for PoonDomingos.

    Args:
        delta (Union[float, List[float], List[List[float]]]): Arg delta for PoonDomingos.
        shape (Sequence[int]): Arg shape for PoonDomingos.
        axes (Sequence[int]): Arg axes for PoonDomingos.

    Returns:
        List[List[List[int]]]: The cut points on each axis (without 0 and 1), in shape \
            (num_deltas, len_axes, num_cuts).
    """
    # For type checking, float works, but for isinstance float does not cover int.
    if isinstance(delta, (float, int)):
        delta = [delta]  # Single delta to list of one delta.
    delta = [  # List of deltas to list of deltas for each axis.
        [delta_i] * len(axes) if isinstance(delta_i, (float, int)) else delta_i for delta_i in delta
    ]

    assert all(
        len(delta_i) == len(axes) for delta_i in delta
    ), "Each delta list must be of same length as axes."
    assert all(
        delta_i_ax >= 1 for delta_i in delta for delta_i_ax in delta_i
    ), "Each delta must be >=1."

    # ANNOTATE: Specify content for empty container.
    cut_points: List[List[List[int]]] = []
    for delta_i in delta:
        cut_pts_i: List[List[int]] = []
        for ax, delta_i_ax in zip(axes, delta_i):
            num_cuts = int((shape[ax] - 1) // delta_i_ax)
            cut_pts_i.append([int((j + 1) * delta_i_ax) for j in range(num_cuts)])
        cut_points.append(cut_pts_i)
    return cut_points


# TODO: too-complex,too-many-locals. how to solve?
# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name,too-complex,too-many-locals
def PoonDomingos(
    shape: Sequence[int],
    *,
    delta: Union[float, List[float], List[List[float]]],
    axes: Optional[Sequence[int]] = None,
    max_depth: Optional[int] = None,
) -> RegionGraph:
    """Construct a RG with the Poon-Domingos structure.

    See:
        Sum-Product Networks: A New Deep Architecture.
        Hoifung Poon, Pedro Domingos.
        UAI 2011.

    Args:
        shape (Sequence[int]): The shape of the hypercube for the variables.
        delta (Union[float, List[float], List[List[float]]]): The deltas to cut the hypercube, can \
            be: a single cut delta for all axes, a list for all axes, a list of list for each \
            axis. If the last case, all inner lists must have the same length as axes.
        axes (Optional[Sequence[int]], optional): The axes to cut. Default means all axes. \
            Defaults to None.
        max_depth (Optional[int], optional): The max depth for cutting, omit for unconstrained. \
            Defaults to None.

    Returns:
        RegionGraph: The PD RG.
    """
    if axes is None:
        axes = tuple(range(len(shape)))
    cut_points = _parse_poon_domingos_delta(delta, shape, axes)

    if max_depth is None:
        max_depth = sum(shape) + 1  # Larger than every possible cuts

    # The list of region and partition nodes
    nodes: List[RegionGraphNode] = []

    # The in-degree connections of region/partition nodes
    in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = defaultdict(list)

    # A map from scopes to the corresponding region nodes
    scope_region: Dict[Scope, RegionNode] = {}

    # An object mapping hypercube selections to region scopes
    hypercube_to_scope = HypercubeToScope(shape)

    # ANNOTATE: Specify content for empty container.
    queue: Deque[HyperCube] = deque()
    depth_dict: Dict[HyperCube, int] = {}  # Also serve as a "visited" set.

    cur_hypercube = ((0,) * len(shape), tuple(shape))
    root_scope = hypercube_to_scope[cur_hypercube]
    root = RegionNode(root_scope)
    nodes.append(root)
    scope_region[root_scope] = root
    queue.append(cur_hypercube)
    depth_dict[cur_hypercube] = 0

    # DISABLE: This is considered a constant.
    SENTINEL = ((-1,) * len(shape), (-1,) * len(shape))  # pylint: disable=invalid-name

    def cut_hypercube_(
        hypercube: HyperCube,
        axis: int,
        cut_points: Union[int, Sequence[int]],
        hypercube_to_scope: HypercubeToScope,
    ) -> List[HyperCube]:
        """Cut a hypercube along given axis at given cut points, and add corresponding regions to RG.

        Args:
            hypercube (HyperCube): The hypercube to cut.
            axis (int): The axis to cut along.
            cut_points (Union[int, Sequence[int]]): The points to cut at, can be a single number for a \
                single cut.
            hypercube_to_scope (HypercubeToScope): The mapping from hypercube to scope.

        Returns:
            List[HyperCube]: The sub-hypercubes that needs further cutting.
        """
        if isinstance(cut_points, int):
            cut_points = [cut_points]

        # Here there should be one found.
        rgn = scope_region.get(hypercube_to_scope[hypercube], None)
        if rgn is None:
            rgn = RegionNode(hypercube_to_scope[hypercube])
            nodes.append(rgn)
        point1, point2 = hypercube
        assert all(
            point1[axis] < cut_point < point2[axis] for cut_point in cut_points
        ), "Cut point out of bounds."

        # ANNOTATE: Specify content for empty container.
        cut_points = [point1[axis]] + sorted(cut_points) + [point2[axis]]
        hypercubes: List[HyperCube] = []
        region_nodes: List[RegionNode] = []
        for cut_l, cut_r in zip(cut_points[:-1], cut_points[1:]):
            # FUTURE: for cut_l, cut_r in itertools.pairwise(cut_points) in 3.10
            point_l, point_r = list(point1), list(point2)  # Must convert to list to modify.
            point_l[axis], point_r[axis] = cut_l, cut_r
            hypercube = tuple(point_l), tuple(point_r)
            hypercubes.append(hypercube)
            rgn_hypercube = scope_region.get(hypercube_to_scope[hypercube], None)
            if rgn_hypercube is None:
                rgn_hypercube = RegionNode(hypercube_to_scope[hypercube])
                nodes.append(rgn_hypercube)
            region_nodes.append(rgn_hypercube)

        # Add partitioning
        ptn = PartitionNode(rgn.scope)
        nodes.append(ptn)
        in_nodes[rgn].append(ptn)
        in_nodes[ptn] = region_nodes
        return hypercubes

    def queue_popleft() -> HyperCube:
        """Wrap queue.popleft() with sentinel value.

        Returns:
            HyperCube: The result of queue.popleft(), or SENTINEL when queue is exhausted.
        """
        try:
            return queue.popleft()
        except IndexError:
            return SENTINEL

    for cur_hypercube in iter(queue_popleft, SENTINEL):
        if depth_dict[cur_hypercube] > max_depth:
            continue

        found_cut = False
        for cut_pts_i in cut_points:
            for ax, cut_pts_i_ax in zip(axes, cut_pts_i):
                for cut_pt in cut_pts_i_ax:
                    if not cur_hypercube[0][ax] < cut_pt < cur_hypercube[1][ax]:
                        continue
                    found_cut = True
                    hypercubes = cut_hypercube_(cur_hypercube, ax, cut_pt, hypercube_to_scope)
                    hypercubes = [cube for cube in hypercubes if cube not in depth_dict]
                    queue.extend(hypercubes)
                    for cube in hypercubes:
                        depth_dict[cube] = depth_dict[cur_hypercube] + 1

            if found_cut:
                break

    return RegionGraph(nodes, in_nodes, outputs=[root])
