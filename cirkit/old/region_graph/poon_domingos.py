import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .region_graph import RegionGraph
from .rg_node import PartitionNode, RegionNode
from .utils import HyperCube, HypercubeScopeCache

# TODO: rework docstrings


def _cut_hypercube(hypercube: HyperCube, axis: int, pos: int) -> Tuple[HyperCube, HyperCube]:
    """Cuts a discrete hypercube into two sub-hypercubes.

    Helper routine for Poon-Domingos (PD) structure.

    A hypercube is represented as a tuple (l, r), where l and r are tuples of \
        ints, representing discrete coordinates. \
        For example ((0, 0), (10, 8)) represents a 2D hypercube (rectangle) whose \
        upper-left coordinate is (0, 0) and its \
        lower-right coordinate (10, 8). Note that upper, lower, left, right are \
        arbitrarily assigned terms here.

    This function cuts a given hypercube in a given axis at a given position.

    :param hypercube: coordinates of the hypercube ((tuple of ints, tuple of ints))
    :param axis: in which axis to cut (int)
    :param pos: at which position to cut (int)
    :return: coordinates of the two hypercubes
    """
    assert hypercube[0][axis] < pos < hypercube[1][axis]

    coord_rigth = list(hypercube[1])
    coord_rigth[axis] = pos
    child1 = (hypercube[0], tuple(coord_rigth))

    coord_left = list(hypercube[0])
    coord_left[axis] = pos
    child2 = (tuple(coord_left), hypercube[1])

    return child1, child2


def _get_region_nodes_by_scope(graph: RegionGraph, scope: Iterable[int]) -> List[RegionNode]:
    """Get `RegionNode`s with a specific scope.

    Args:
        graph (nx.DiGraph): The region graph to find in.
        scope (Iterable[int]): The scope to find.

    Returns:
        List[RegionNode]: The `RegionNode`s found with the scope
    """
    scope = set(scope)
    return [n for n in graph.region_nodes if n.scope == scope]


# TODO: refactor
# pylint: disable-next=too-complex,too-many-locals,too-many-branches,invalid-name
def PoonDomingos(
    shape: Sequence[int],
    delta: Union[float, int, List[Union[float, int]], List[List[Union[float, int]]]],
    axes: Optional[Sequence[int]] = None,
    max_split_depth: Optional[int] = None,
) -> RegionGraph:
    """Get a RG in PD.

    The PD structure generates a PC structure for random variables which can \
        be naturally arranged on discrete grids, like images.

    Ref:
        Sum-Product Networks: A New Deep Architecture
        Hoifung Poon, Pedro Domingos
        UAI 2011

    This function implements PD structure, generalized to grids of arbitrary dimensions: 1D
    (e.g. sequences), 2D (e.g. images), 3D (e.g. video), ...
    Here, these grids are called hypercubes, and represented via two coordinates, corresponding
    to the corner with lowest coordinates and corner with largest coordinates.
    For example,
        ((1,), (5,)) is a 1D hypercube ranging from 1 to 5
        ((2,3), (7,7)) is a 2D hypercube ranging from 2 to 7 for the first axis, and from 3 to 7
        for the second axis.

    Each coordinate in a hypercube/grid corresponds to a random variable (RVs).
    The argument shape determines the overall hypercube. For example, shape = (28, 28)
    corresponds to a 2D hypercube containing 28*28 = 784 random variables.
    This would be appropriate, for example, to model MNIST images. The overall hypercube
    has coordinates ((0, 0), (28, 28)). We index the RVs with a linear index,
    which toggles fastest for higher axes. For example, a (5, 5) hypercube gets linear indices
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]  ->   (0, 1, 2, 3, ..., 21, 22, 23, 24)

    Sum nodes and leaves in PCs correspond to sub-hypercubes, and the corresponding unrolled linear
    indices serve as scope for these PC nodes. For example, the sub-hypercube
    ((1, 2), (4, 5)) of the (5, 5) hypercube above gets scope
        [[ 7  8  9]
         [12 13 14]
         [17 18 19]]   ->   (7, 8, 9, 12, 13, 14, 17, 18, 19)

    The PD structure starts with a single sum node corresponding to the overall hypercube. Then,
    it recursively splits the hypercube using axis-aligned cuts. A cut corresponds to a product
    node, and the split parts correspond again to sums or leaves.
    Regions are split in several ways, by displacing the cut point by some delta. Note that
    sub-hypercubes can typically be obtained by different ways to cut. For example, splitting

        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]

    into

    [[ 0  1]    |   [[ 2  3  4]
     [ 5  6]    |    [ 7  8  9]
     [10 11]    |    [12 13 14]
     [15 16]    |    [17 18 19]
     [20 21]]   |    [22 23 24]]

    and then splitting the left hypercube into

    [[ 0  1]
     [ 5  6]]
    ----------
    [[10 11]
     [15 16]
     [20 21]]

    Gives us the hypercube with scope (0, 1, 5, 6). Alternatively, we could also cut

    [[0 1 2 3 4]
     [5 6 7 8 9]]
    -------------------
    [[10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]

    and then cut the upper hypercube into

    [[0 1]   |  [[2 3 4]
     [5 6]]  |   [7 8 9]]

    which again gives us the hypercube with scope (0, 1, 5, 6). Thus, we obtained the same
    hypercube, (0, 1, 5, 6), via two (in in general more) alternative cutting processes.
    What is important is that this hypercube is *not duplicated*, but we re-use it when we
    re-encounter it. In PCs, this means that the sum node associated with (0, 1, 5, 6) becomes
    a shared child of many product nodes. This sharing yields PC structures, which resembles a bit a
    convolutional structures. Thus, the PD structure has arguably a suitable inductive bias for
    array-shaped data.

    The displacement of the cutting points is governed via argument delta.
    We can also specify multiple deltas, and also different delta values for different axes.
    We first compute all cutting points on the overall hypercube, for each specified delta and
    each axis. When we encounter a hypercube in the recursive splitting process, we consider
    each axis and split it on all cutting points corresponding to the coarsest delta.

    :param shape: shape of the overall hypercube (tuple of ints)
    :param delta: determines the displacement of cutting points.
                numerical: a single displacement value, applied to all axes.
                list of numerical: several displacement values, applied to all axes.
                list of list of numerical: several displacement values, specified for each
                individual axis. In this case, the outer list must be of same length as axes.
    :param axes: which axes are subject to cutting? (tuple of ints)
                For example, if shape = (5, 5) (2DGrid), then axes = (0,) means that we only cut
                along the first axis.
                Can be None, in which case all axes are subject to cutting.
    :param max_split_depth: maximal depth for the recursive split process (int)
    :return: the RG.
    """
    if axes is None:
        axes = tuple(range(len(shape)))
    if max_split_depth is None:
        # TODO: is is correct: depth will not be larger than this
        max_split_depth = sum(shape) + 1
    if isinstance(delta, (float, int)):
        delta = [delta]
    # TODO: how to better handle possible int?
    delta = [
        [deltai] * len(axes) if isinstance(deltai, (float, int)) else deltai for deltai in delta
    ]

    for deltai in delta:
        assert len(deltai) == len(
            axes
        ), "Each delta must either be list of length len(axes), or numeric."
        for deltaij in deltai:
            assert deltaij >= 1, "Any delta must be >= 1."

    shape_to_cut = tuple(shp for ax, shp in enumerate(shape) if ax in axes)

    global_cut_points: List[List[List[int]]] = []
    for deltai in delta:
        cur_global_cur_points: List[List[int]] = []
        for shp, deltaij in zip(shape_to_cut, deltai):
            num_cuts = math.floor((shp - 1) / deltaij)
            cps = [math.ceil((i + 1) * deltaij) for i in range(num_cuts)]
            cur_global_cur_points.append(cps)
        global_cut_points.append(cur_global_cur_points)

    hypercube_to_scope = HypercubeScopeCache()
    hypercube = ((0,) * len(shape), tuple(shape))  # TODO: fit param type
    hypercube_scope = hypercube_to_scope(hypercube, shape)

    root = RegionNode(hypercube_scope)
    graph = RegionGraph()
    graph.add_node(root)

    queue: List[HyperCube] = [hypercube]
    depth_dict = {tuple(hypercube_scope): 0}

    # TODO: refactor for nest block
    while queue:  # pylint: disable=while-used,too-many-nested-blocks
        hypercube = queue.pop(0)
        hypercube_scope = hypercube_to_scope(hypercube, shape)
        # TODO: redundant cast to tuple
        if (depth := depth_dict[tuple(hypercube_scope)]) >= max_split_depth:
            continue

        node = _get_region_nodes_by_scope(graph, hypercube_scope)[0]

        found_cut_on_level = False
        for cur_global_cut_points in global_cut_points:
            for ax_id, axis in enumerate(axes):
                cut_points = [
                    c
                    for c in cur_global_cut_points[ax_id]
                    if hypercube[0][axis] < c < hypercube[1][axis]
                ]
                if len(cut_points) > 0:
                    found_cut_on_level = True

                for idx in cut_points:
                    child_hypercubes = _cut_hypercube(hypercube, axis, idx)
                    child_nodes: List[RegionNode] = []
                    for c_cube in child_hypercubes:
                        c_scope = hypercube_to_scope(c_cube, shape)
                        if not (c_node := _get_region_nodes_by_scope(graph, c_scope)):
                            c_node.append(RegionNode(c_scope))
                            depth_dict[tuple(c_scope)] = depth + 1
                            queue.append(c_cube)
                        child_nodes.append(c_node[0])

                    partition = PartitionNode(node.scope)
                    graph.add_edge(partition, node)
                    for ch_node in child_nodes:
                        graph.add_edge(ch_node, partition)
            if found_cut_on_level:
                break

    return graph
