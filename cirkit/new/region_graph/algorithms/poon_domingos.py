from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Union

from cirkit.new.region_graph.algorithms.utils import HyperCube, HypercubeToScope
from cirkit.new.region_graph.region_graph import RegionGraph
from cirkit.new.region_graph.rg_node import RegionNode
from cirkit.new.utils import Scope

# TODO: test what is constructed here


def _get_region_node_by_scope(graph: RegionGraph, scope: Scope) -> RegionNode:
    """Find a RegionNode with a specific scope in the RG, and construct one if not found.

    Args:
        graph (RegionGraph): The region graph to find in.
        scope (Scope): The scope to find.

    Returns:
        RegionNode: The RegionNode found or constructed.
    """
    # Should found at most one, by PD algorithm.
    region_node = next((node for node in graph.region_nodes if node.scope == scope), None)
    return region_node if region_node is not None else RegionNode(scope)


def _cut_hypercube(
    hypercube: HyperCube,
    axis: int,
    cut_points: Union[int, Sequence[int]],
    hypercube_to_scope: HypercubeToScope,
    graph: RegionGraph,
) -> List[HyperCube]:
    """Cut a hypercube along given axis at given cut points, and add corresponding regions to RG.

    Args:
        hypercube (HyperCube): The hypercube to cut.
        axis (int): The axis to cut along.
        cut_points (Union[int, Sequence[int]]): The points to cut at, can be a single number for a \
            single cut.
        hypercube_to_scope (HypercubeToScope): The mapping from hypercube to scope.
        graph (RegionGraph): The region graph to hold the cut.

    Returns:
        List[HyperCube]: The sub-hypercubes that needs further cutting.
    """
    if isinstance(cut_points, int):
        cut_points = [cut_points]

    # Here there should be one found.
    node = _get_region_node_by_scope(graph, hypercube_to_scope[hypercube])
    point1, point2 = hypercube

    assert all(
        point1[axis] < cut_point < point2[axis] for cut_point in cut_points
    ), "Cut point out of bounds."

    cut_points = [point1[axis]] + sorted(cut_points) + [point2[axis]]

    # ANNOTATE: Specify content for empty container.
    hypercubes: List[HyperCube] = []
    region_nodes: List[RegionNode] = []
    for cut_l, cut_r in zip(cut_points[:-1], cut_points[1:]):
        # FUTURE: for cut_l, cut_r in itertools.pairwise(cut_points) in 3.10
        point_l, point_r = list(point1), list(point2)  # Must convert to list to modify.
        point_l[axis], point_r[axis] = cut_l, cut_r
        hypercube = tuple(point_l), tuple(point_r)
        hypercubes.append(hypercube)
        region_nodes.append(_get_region_node_by_scope(graph, hypercube_to_scope[hypercube]))

    graph.add_partitioning(node, region_nodes)
    return hypercubes


def _parse_delta(
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

    cut_points = _parse_delta(delta, shape, axes)

    if max_depth is None:
        max_depth = sum(shape) + 1  # Larger than every possible cuts

    graph = RegionGraph()
    hypercube_to_scope = HypercubeToScope(shape)
    # ANNOTATE: Specify content for empty container.
    queue: Deque[HyperCube] = deque()
    depth_dict: Dict[HyperCube, int] = {}  # Also serve as a "visited" set.

    cur_hypercube = ((0,) * len(shape), tuple(shape))
    graph.add_node(RegionNode(hypercube_to_scope[cur_hypercube]))
    queue.append(cur_hypercube)
    depth_dict[cur_hypercube] = 0

    # DISABLE: This is considered a constant.
    SENTINEL = ((-1,) * len(shape), (-1,) * len(shape))  # pylint: disable=invalid-name

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
                    hypercubes = _cut_hypercube(
                        cur_hypercube, ax, cut_pt, hypercube_to_scope, graph
                    )
                    hypercubes = [cube for cube in hypercubes if cube not in depth_dict]
                    queue.extend(hypercubes)
                    for cube in hypercubes:
                        depth_dict[cube] = depth_dict[cur_hypercube] + 1

            if found_cut:
                break

    return graph.freeze()
