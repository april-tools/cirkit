import itertools
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import cast

from cirkit.templates.region_graph.algorithms.utils import HyperCube, HypercubeToScope
from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)
from cirkit.utils.scope import Scope


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def PoonDomingos(
    shape: tuple[int, int, int],
    *,
    delta: float | list[float] | list[list[float]],
    max_depth: int | None = None,
) -> RegionGraph:
    r"""Constructs a region graph with the Poon-Domingos structure.

    See:
        Sum-Product Networks: A New Deep Architecture.
        Hoifung Poon, Pedro Domingos.
        UAI 2011.

    Args:
        shape: The image shape $(C, H, W)$, where $H$ is the height, $W$ is the width,
            and $C$ is the number of channels.
        delta: The deltas to cut the hypercube, can
            be: a single cut delta for all axes, a list for all axes, a list of list for each
            axis. If the last case, all inner lists must have the same length as axes.
        max_depth: The max depth for cutting, omit for unconstrained.
            Defaults to None.

    Returns:
        RegionGraph: The Poon-Domingos region grpah.
    """
    axes = (1, 2)  # The axes to cut, i.e., the height and width axes.
    cut_points = _parse_poon_domingos_delta(delta, shape, axes)

    if max_depth is None:
        max_depth = sum(shape) + 1  # Larger than every possible cuts

    # The list of region and partition nodes
    nodes: list[RegionGraphNode] = []

    # The in-degree connections of region/partition nodes
    in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = defaultdict(list)

    # A map from scopes to the corresponding region nodes
    scope_region: dict[Scope, RegionNode] = {}

    # An object mapping hypercube selections to region scopes
    hypercube_to_scope = HypercubeToScope(shape)

    # ANNOTATE: Specify content for empty container.
    queue: deque[HyperCube] = deque()
    depth_dict: dict[HyperCube, int] = {}  # Also serve as a "visited" set.

    cur_hypercube: tuple[tuple[int, ...], tuple[int, ...]] = ((0,) * len(shape), shape)
    root_scope = hypercube_to_scope[cur_hypercube]
    root = RegionNode(root_scope)
    nodes.append(root)
    scope_region[root_scope] = root
    queue.append(cur_hypercube)
    depth_dict[cur_hypercube] = 0

    # DISABLE: This is considered a constant
    SENTINEL = ((-1,) * len(shape), (-1,) * len(shape))  # pylint: disable=invalid-name

    def cut_hypercube_(
        hypercube: HyperCube,
        axis: int,
        cut_points: int | Sequence[int],
        hypercube_to_scope: HypercubeToScope,
    ) -> list[HyperCube]:
        """Cut a hypercube along given axis at given cut points, and add corresponding regions to
         the region graph.

        Args:
            hypercube (HyperCube): The hypercube to cut.
            axis (int): The axis to cut along.
            cut_points (Union[int, Sequence[int]]): The points to cut at, can be a single number
                for a single cut.
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
            scope_region[hypercube_to_scope[hypercube]] = rgn
        point1, point2 = hypercube
        assert all(
            point1[axis] < cut_point < point2[axis] for cut_point in cut_points
        ), "Cut point out of bounds."

        # ANNOTATE: Specify content for empty container.
        cut_points = [point1[axis]] + sorted(cut_points) + [point2[axis]]
        hypercubes: list[HyperCube] = []
        region_nodes: list[RegionNode] = []
        for cut_l, cut_r in itertools.pairwise(cut_points):
            point_l, point_r = list(point1), list(point2)  # Must convert to list to modify.
            point_l[axis], point_r[axis] = cut_l, cut_r
            hypercube = tuple(point_l), tuple(point_r)
            hypercubes.append(hypercube)
            rgn_hypercube = scope_region.get(hypercube_to_scope[hypercube], None)
            if rgn_hypercube is None:
                rgn_hypercube = RegionNode(hypercube_to_scope[hypercube])
                nodes.append(rgn_hypercube)
                scope_region[hypercube_to_scope[hypercube]] = rgn_hypercube
            region_nodes.append(rgn_hypercube)

        # Add partitioning
        ptn = PartitionNode(rgn.scope)
        nodes.append(ptn)
        in_nodes[rgn].append(ptn)
        in_nodes[ptn] = cast(list[RegionGraphNode], region_nodes)
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


def _parse_poon_domingos_delta(
    delta: float | list[float] | list[list[float]], shape: Sequence[int], axes: Sequence[int]
) -> list[list[list[int]]]:
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
    cut_points: list[list[list[int]]] = []
    for delta_i in delta:
        cut_pts_i: list[list[int]] = []
        for ax, delta_i_ax in zip(axes, delta_i):
            num_cuts = int((shape[ax] - 1) // delta_i_ax)
            cut_pts_i.append([int((j + 1) * delta_i_ax) for j in range(num_cuts)])
        cut_points.append(cut_pts_i)
    return cut_points
