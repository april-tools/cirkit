from itertools import count
from typing import List

import networkx as nx
import numpy as np


class EiNetAddress:
    """
    Address of a PC node to its EiNet implementation.

    In EiNets, each layer implements a tensor of log-densities of shape
        (batch_size, vector_length, num_nodes)
    All DistributionVector's, which are either vectors of leaf distributions (exponential families)
    or vectors of sum nodes, uniquely correspond to some slice of the log-density tensor of some
    layer, where we slice the last axis.

    EiNetAddress stores the "address" of the implementation in the EinsumNetwork.
    """

    def __init__(self, layer=None, idx=None, replica_idx=None):
        """
        :param layer: which layer implements this node?
        :param idx: which index does the node have in the layers log-density tensor?
        :param replica_idx: this is solely for the input layer -- see ExponentialFamilyArray and
        FactorizedLeafLayer.
        These two layers implement all leaves in parallel. To this end we need "enough leaves",
        which is achieved to make a sufficiently large "block" of input distributions.
        The replica_idx indicates in which slice of the ExponentialFamilyArray a leaf is
        represented.
        """
        self.layer = layer
        self.idx = idx
        self.replica_idx = replica_idx


class RegionNode:
    """
    Represents either a vectorized leaf or a vectorized sum node in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    # we assign each object a unique id.
    _id_counter = count(0)

    def __init__(self, scope):
        """
        :param scope: the scope of this node
        """
        self.scope = tuple(sorted(scope))
        self.num_dist = None
        self.einet_address = EiNetAddress()
        self.id = next(self._id_counter)

    def __repr__(self) -> str:
        return self.scope.__repr__()

    def __lt__(self, other):
        if type(other) == PartitionNode:
            return True
        else:
            return (self.scope, self.id) < (other.scope, other.id)


class PartitionNode:
    """
    Represents a (cross-)product in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    # we assign each object a unique id.
    _id_counter = count(0)

    def __init__(self, scope):
        self.scope = tuple(sorted(scope))
        self.id = next(self._id_counter)

    def __repr__(self):
        return "Prod for " + self.scope.__repr__()

    def __lt__(self, other):
        if type(other) == RegionNode:
            return False
        else:
            return (self.scope, self.id) < (other.scope, other.id)


def check_if_is_partition(X, P):
    """
    Checks if P represents a partition of X.

    :param X: some iterable representing a set of objects.
    :param P: some iterable of iterables, representing a set of sets.
    :return: True of P is a partition of X
                 i) union over P is X
                 ii) sets in P are non-overlapping
    """
    P_as_sets = [set(p) for p in P]
    union = set().union(*[set(p) for p in P_as_sets])
    non_overlapping = len(union) == sum([len(p) for p in P_as_sets])
    return set(X) == union and non_overlapping


def check_graph(graph: nx.DiGraph) -> (bool, str):
    """
    Check if a graph satisfies our requirements for PC graphs.

    :param graph:
    :return: True/False (bool), string description
    """

    contains_only_PC_nodes = all(
        [type(n) == RegionNode or type(n) == PartitionNode for n in graph.nodes()]
    )

    is_DAG = nx.is_directed_acyclic_graph(graph)
    is_connected = nx.is_connected(graph.to_undirected())

    sums = get_sums(graph)
    products = get_products(graph)

    products_one_parents = all([len(list(graph.predecessors(p))) == 1 for p in products])
    products_two_children = all([len(list(graph.successors(p))) == 2 for p in products])

    sum_to_products = all(
        [all([type(p) == PartitionNode for p in graph.successors(s)]) for s in sums]
    )
    product_to_dist = all(
        [all([type(s) == RegionNode for s in graph.successors(p)]) for p in products]
    )
    alternating = sum_to_products and product_to_dist

    proper_scope = all([len(n.scope) == len(set(n.scope)) for n in graph.nodes()])
    smooth = all([all([p.scope == s.scope for p in graph.successors(s)]) for s in sums])
    decomposable = all(
        [check_if_is_partition(p.scope, [s.scope for s in graph.successors(p)]) for p in products]
    )

    check_passed = (
        contains_only_PC_nodes
        and is_DAG
        and is_connected
        and products_one_parents
        and products_two_children
        and alternating
        and proper_scope
        and smooth
        and decomposable
    )

    msg = ""
    if check_passed:
        msg += "Graph check passed.\n"
    if not contains_only_PC_nodes:
        msg += "Graph does not only contain DistributionVector or Product nodes.\n"
    if not is_connected:
        msg += "Graph not connected.\n"
    if not products_one_parents:
        msg += "Products do not have exactly one parent.\n"
    if not products_two_children:
        msg += "Products do not have exactly two children.\n"
    if not alternating:
        msg += "Graph not alternating.\n"
    if not proper_scope:
        msg += "Scope is not proper.\n"
    if not smooth:
        msg += "Graph is not smooth.\n"
    if not decomposable:
        msg += "Graph is not decomposable.\n"

    return check_passed, msg.rstrip()


def get_roots(graph):
    return [n for n, d in graph.in_degree() if d == 0]


def get_sums(graph):
    return [n for n, d in graph.out_degree() if d > 0 and type(n) == RegionNode]


def get_products(graph):
    return [n for n in graph.nodes() if type(n) == PartitionNode]


def get_leaves(graph):
    return [n for n, d in graph.out_degree() if d == 0]


def get_distribution_nodes_by_scope(graph, scope):
    scope = tuple(sorted(scope))
    return [n for n in graph.nodes if type(n) == RegionNode and n.scope == scope]


def partition_on_node(graph, node, scope_partition):
    """
    Helper routine to extend the graph.

    Takes a node and adds a new product child to it. Furthermore, as children of the product, it adds new
    DistributionVector nodes with scopes as prescribed in scope_partition (must be a proper partition of the node's
    scope).

    :param graph: PC graph (DiGraph)
    :param node: node in the graph (DistributionVector)
    :param scope_partition: partition of the node's scope
    :return: the product and a list if the product's children
    """

    if not check_if_is_partition(node.scope, scope_partition):
        raise AssertionError("Not a partition.")

    product = PartitionNode(node.scope)
    graph.add_edge(node, product)
    product_children = [RegionNode(scope) for scope in scope_partition]
    for c in product_children:
        graph.add_edge(product, c)

    return product, product_children


def randomly_partition_on_node(graph, node, num_parts=2, proportions=None, rand_state=None):
    """
    Calls partition_on_node with a random partition -- used for random binary trees (RAT-SPNs).

    :param graph: PC graph (DiGraph)
    :param node: node in the graph (DistributionVector)
    :param num_parts: number of parts in the partition (int)
    :param proportions: split proportions (list of numbers)
    :param rand_state: numpy random_state to use for random split; if None the default numpy random
    state is used
    :return: the product and a list of the products children
    """
    if proportions is not None:
        if num_parts is None:
            num_parts = len(proportions)
        else:
            if len(proportions) != num_parts:
                raise AssertionError("proportions should have num_parts elements.")
        proportions = np.array(proportions).astype(np.float64)
    else:
        proportions = np.ones(num_parts).astype(np.float64)

    if num_parts > len(node.scope):
        raise AssertionError(
            "Cannot split scope of length {} into {} parts.".format(len(node.scope), num_parts)
        )

    proportions /= proportions.sum()
    if rand_state is not None:
        permutation = list(rand_state.permutation(list(node.scope)))
    else:
        permutation = list(np.random.permutation(list(node.scope)))

    child_indices = []
    for p in range(num_parts):
        p_len = int(np.round(len(permutation) * proportions[0]))
        p_len = min(max(p_len, 1), p + 1 + len(permutation) - num_parts)
        child_indices.append(permutation[0:p_len])
        permutation = permutation[p_len:]
        proportions = proportions[1:]
        proportions /= proportions.sum()

    return partition_on_node(graph, node, child_indices)


def random_binary_trees(num_var, depth, num_repetitions):
    """
    Generate a PC graph via several random binary trees -- RAT-SPNs.

    See
        Random sum-product networks: A simple but effective approach to probabilistic deep learning
        Robert Peharz, Antonio Vergari, Karl Stelzner, Alejandro Molina, Xiaoting Shao,
        Martin Trapp, Kristian Kersting, Zoubin Ghahramani
        UAI 2019

    :param num_var: number of random variables (int)
    :param depth: splitting depth (int)
    :param num_repetitions: number of repetitions (int)
    :return: generated graph (DiGraph)
    """
    graph = nx.DiGraph()
    root = RegionNode(range(num_var))
    graph.add_node(root)

    for repetition in range(num_repetitions):
        cur_nodes = [root]
        for d in range(depth):
            child_nodes = []
            for node in cur_nodes:
                _, cur_child_nodes = randomly_partition_on_node(graph, node, 2)
                child_nodes += cur_child_nodes
            cur_nodes = child_nodes
        for node in cur_nodes:
            node.einet_address.replica_idx = repetition

    return graph


def cut_hypercube(hypercube, axis, pos):
    """
    Helper routine for Poon-Domingos (PD) structure. Cuts a discrete hypercube into two sub-hypercubes.

    A hypercube is represented as a tuple (l, r), where l and r are tuples of ints, representing discrete coordinates.
    For example ((0, 0), (10, 8)) represents a 2D hypercube (rectangle) whose upper-left coordinate is (0, 0) and its
    lower-right coordinate (10, 8). Note that upper, lower, left, right are arbitrarily assigned terms here.

    This function cuts a given hypercube in a given axis at a given position.

    :param hypercube: coordinates of the hypercube ((tuple of ints, tuple of ints))
    :param axis: in which axis to cut (int)
    :param pos: at which position to cut (int)
    :return: coordinates of the two hypercubes
    """
    if pos <= hypercube[0][axis] or pos >= hypercube[1][axis]:
        raise AssertionError

    coord_rigth = list(hypercube[1])
    coord_rigth[axis] = pos
    child1 = (hypercube[0], tuple(coord_rigth))

    coord_left = list(hypercube[0])
    coord_left[axis] = pos
    child2 = (tuple(coord_left), hypercube[1])

    return child1, child2


class HypercubeToScopeCache:
    """
    Helper class for Poon-Domingos (PD) structure. Represents a function cache, mapping hypercubes to their unrolled
    scope.

    For example consider the hypercube ((0, 0), (4, 5)), which is a rectangle with 4 rows and 5 columns. We assign
    linear indices to the elements in this rectangle as follows:
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
    Similarly, we assign linear indices to higher-dimensional hypercubes, where higher axes toggle faster than lower
    axes. The scope of sub-hypercubes are just the unrolled linear indices. For example, for the rectangle above,
    the sub-rectangle ((1, 2), (4, 5)) has scope (7, 8, 9, 12, 13, 14, 17, 18, 19).

    This class just represents a cached mapping from hypercubes to their scopes.
    """

    def __init__(self):
        self._hyper_cube_to_scope = {}

    def __call__(self, hypercube, shape):
        if hypercube in self._hyper_cube_to_scope:
            return self._hyper_cube_to_scope[hypercube]

        x1 = hypercube[0]
        x2 = hypercube[1]

        if len(x1) != len(x2) or len(x1) != len(shape):
            raise AssertionError
        for i in range(len(shape)):
            if x1[i] < 0 or x2[i] > shape[i]:
                raise AssertionError

        scope = np.zeros(tuple(x2[i] - x1[i] for i in range(len(shape))), np.int64)
        f = 1
        for i, c in enumerate(reversed(range(len(shape)))):
            range_to_add = f * np.array(range(x1[c], x2[c]), np.int64)
            scope += np.reshape(range_to_add, (len(range_to_add),) + i * (1,))
            f *= shape[c]

        scope = tuple(scope.reshape(-1))
        self._hyper_cube_to_scope[hypercube] = scope
        return scope


def poon_domingos_structure(shape, delta, axes=None, max_split_depth=None):
    """
    The PD structure was proposed in
        Sum-Product Networks: A New Deep Architecture
        Hoifung Poon, Pedro Domingos
        UAI 2011
    and generates a PC structure for random variables which can be naturally arranged on discrete
    grids, like images.

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
    :return: PC graph (DiGraph)
    """
    shape = tuple(shape)
    if any([type(s) != int for s in shape]):
        raise TypeError("Elements in shape must be ints.")

    if axes is None:
        axes = list(range(len(shape)))

    try:
        delta = list(delta)
    except TypeError:
        delta = [delta]

    for c in range(len(delta)):
        try:
            delta[c] = list(delta[c])
            if len(delta[c]) != len(axes):
                raise AssertionError(
                    "Each delta must either be list of length len(axes), or numeric."
                )
        except TypeError:
            delta[c] = [float(delta[c])] * len(axes)

    if any([dd < 1.0 for d in delta for dd in d]):
        raise AssertionError("Any delta must be >= 1.0.")

    sub_shape = tuple(s for c, s in enumerate(shape) if c in axes)
    global_cut_points = []
    for dd in delta:
        cur_global_cur_points = []
        for s, d in zip(sub_shape, dd):
            num_cuts = int(np.floor(float(s - 1) / d))
            cps = [int(np.ceil((i + 1) * d)) for i in range(num_cuts)]
            cur_global_cur_points.append(cps)
        global_cut_points.append(cur_global_cur_points)

    hypercube_to_scope = HypercubeToScopeCache()
    hypercube = ((0,) * len(shape), shape)
    hypercube_scope = hypercube_to_scope(hypercube, shape)

    graph = nx.DiGraph()
    root = RegionNode(hypercube_scope)
    graph.add_node(root)

    Q = [hypercube]
    depth_dict = {hypercube_scope: 0}

    while Q:
        hypercube = Q.pop(0)
        hypercube_scope = hypercube_to_scope(hypercube, shape)
        depth = depth_dict[hypercube_scope]
        if max_split_depth is not None and depth >= max_split_depth:
            continue

        node = get_distribution_nodes_by_scope(graph, hypercube_scope)
        if len(node) != 1:
            raise AssertionError("Node not found or duplicate.")
        node = node[0]

        found_cut_on_level = False
        for cur_global_cut_points in global_cut_points:
            if found_cut_on_level:
                break
            for ac, axis in enumerate(axes):
                cut_points = [
                    c
                    for c in cur_global_cut_points[ac]
                    if hypercube[0][axis] < c < hypercube[1][axis]
                ]
                if len(cut_points) > 0:
                    found_cut_on_level = True

                for idx in cut_points:
                    child_hypercubes = cut_hypercube(hypercube, axis, idx)
                    child_nodes = []
                    for c_cube in child_hypercubes:
                        c_scope = hypercube_to_scope(c_cube, shape)
                        c_node = get_distribution_nodes_by_scope(graph, c_scope)
                        if len(c_node) > 1:
                            raise AssertionError("Duplicate node.")
                        if len(c_node) == 1:
                            c_node = c_node[0]
                        else:
                            c_node = RegionNode(c_scope)
                            depth_dict[c_scope] = depth + 1
                            Q.append(c_cube)
                        child_nodes.append(c_node)

                    product = PartitionNode(node.scope)
                    graph.add_edge(node, product)
                    for c_node in child_nodes:
                        graph.add_edge(product, c_node)

    for node in get_leaves(graph):
        node.einet_address.replica_idx = 0

    return graph


def topological_layers(graph):
    """
    Arranging the PC graph in topological layers -- see Algorithm 1 in the paper.

    :param graph: the PC graph (DiGraph)
    :return: list of layers, alternating between DistributionVector and Product layers (list of lists of nodes).
    """
    visited_nodes = set()
    layers = []

    sums = list(sorted(get_sums(graph)))
    products = list(sorted(get_products(graph)))
    leaves = list(sorted(get_leaves(graph)))

    num_internal_nodes = len(sums) + len(products)

    while len(visited_nodes) != num_internal_nodes:
        sum_layer = [
            s
            for s in sums
            if s not in visited_nodes and all([p in visited_nodes for p in graph.predecessors(s)])
        ]
        sum_layer = sorted(sum_layer)
        layers.insert(0, sum_layer)
        visited_nodes.update(sum_layer)

        product_layer = [
            p
            for p in products
            if p not in visited_nodes and all([s in visited_nodes for s in graph.predecessors(p)])
        ]
        product_layer = sorted(product_layer)
        layers.insert(0, product_layer)
        visited_nodes.update(product_layer)

    layers.insert(0, leaves)
    return layers


def plot_graph(graph):
    """
    Plots the PC graph.

    :param graph: the PC graph (DiGraph)
    :return: None
    """
    pos = {}
    layers = topological_layers(graph)
    for i, layer in enumerate(layers):
        for j, item in enumerate(layer):
            pos[item] = np.array([float(j) - 0.25 + 0.5 * np.random.rand(), float(i)])

    distributions = [n for n in graph.nodes if type(n) == RegionNode]
    products = [n for n in graph.nodes if type(n) == PartitionNode]
    node_sizes = [3 + 10 * i for i in range(len(graph))]

    nx.draw_networkx_nodes(graph, pos, distributions, node_shape="p")
    nx.draw_networkx_nodes(graph, pos, products, node_shape="^")
    nx.draw_networkx_edges(graph, pos, node_size=node_sizes, arrowstyle="->", arrowsize=10, width=2)


"""
# run to see some usage examples
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    graph = random_binary_trees(7, 2, 3)
    _, msg = check_graph(graph)
    print(msg)

    plt.figure(1)
    plt.clf()
    plt.title("Random binary tree (RAT-SPN)")
    plot_graph(graph)
    plt.show()

    print()

    graph = poon_domingos_structure((3, 3), delta=1, max_split_depth=None)
    _, msg = check_graph(graph)
    print(msg)
    plt.figure(1)
    plt.clf()
    plt.title("Poon-Domingos Structure")
    plot_graph(graph)
    plt.show()

"""


def quad_tree_graph(width: int, height: int) -> nx.DiGraph:
    assert width == height and width > 0

    shape = (width, height)
    graph = nx.DiGraph()

    hypercube_to_scope = HypercubeToScopeCache()

    # list of lists of hypercubes
    buffer = [[] for _ in range(width)]

    # Add Leaves
    for i in range(width):
        buffer[i] = []
        for j in range(height):
            hypercube = ((i, j), (i + 1, j + 1))

            c_scope = hypercube_to_scope(hypercube, shape)
            c_node = RegionNode(c_scope)
            graph.add_node(c_node)
            buffer[i].append(c_node)

    lr_choice = 0  # random.choice([0, 1])
    td_choice = 0  # random.choice([0, 1])

    old_buffer_width = width
    old_buffer_height = height
    old_buffer = buffer

    while old_buffer_width != 1 and old_buffer_height != 1:
        buffer_width = (
            old_buffer_width // 2 if old_buffer_width % 2 == 0 else old_buffer_width // 2 + 1
        )
        buffer_height = (
            old_buffer_height // 2 if old_buffer_height % 2 == 0 else old_buffer_height // 2 + 1
        )
        buffer = [[] for _ in range(buffer_width)]

        for i in range(buffer_width):
            buffer[i] = [[] for _ in range(buffer_height)]
            for j in range(buffer_height):
                regions = square_from_buffer(old_buffer, 2 * i + lr_choice, 2 * j + td_choice)

                if len(regions) == 1:
                    buffer[i][j] = regions[0]
                elif len(regions) == 2:
                    buffer[i][j] = merge_2_regions(regions, graph)
                elif len(regions) == 4:
                    buffer[i][j] = merge_4_regions(regions, graph)
                else:
                    raise AssertionError("Invalid number of regions")

        old_buffer = buffer
        old_buffer_width = buffer_width
        old_buffer_height = buffer_height

    for node in get_leaves(graph):
        node.einet_address.replica_idx = 0

    check, msg = check_graph(graph)
    assert check
    print(msg)

    return graph


def merge_2_regions(regions: List[RegionNode], graph: nx.DiGraph) -> RegionNode:
    assert len(regions) == 2

    scope = list(set(sorted(regions[0].scope + regions[1].scope)))
    p = PartitionNode(scope)
    d = RegionNode(scope)

    graph.add_edge(p, regions[0])
    graph.add_edge(p, regions[1])
    graph.add_edge(d, p)

    return d


def merge_4_regions(regions: List[RegionNode], graph: nx.DiGraph) -> RegionNode:
    assert len(regions) == 4

    # regions have to have TL, TR, BL, BR
    # MERGE TL & TR, BL & BR
    tscope = list(set(sorted(regions[0].scope + regions[1].scope)))
    bscope = list(set(sorted(regions[2].scope + regions[3].scope)))
    t_p = PartitionNode(tscope)
    t_d = RegionNode(tscope)
    b_p = PartitionNode(bscope)
    b_d = RegionNode(bscope)

    graph.add_edge(t_p, regions[0])
    graph.add_edge(t_p, regions[1])
    graph.add_edge(t_d, t_p)
    graph.add_edge(b_p, regions[2])
    graph.add_edge(b_p, regions[3])
    graph.add_edge(b_d, b_p)

    # MERGE T & B
    whole_scope = list(set(sorted(t_d.scope + b_d.scope)))
    horiz_p = PartitionNode(whole_scope)
    graph.add_edge(horiz_p, t_d)
    graph.add_edge(horiz_p, b_d)

    # MERGE TL & BL, TR & BR
    lscope = list(set(sorted(regions[0].scope + regions[2].scope)))
    rscope = list(set(sorted(regions[1].scope + regions[3].scope)))
    l_p = PartitionNode(lscope)
    l_d = RegionNode(lscope)
    r_p = PartitionNode(rscope)
    r_d = RegionNode(rscope)

    graph.add_edge(l_p, regions[0])
    graph.add_edge(l_p, regions[2])
    graph.add_edge(l_d, l_p)
    graph.add_edge(r_p, regions[1])
    graph.add_edge(r_p, regions[3])
    graph.add_edge(r_d, r_p)

    # MERGE T & B
    assert whole_scope == list(set(sorted(l_d.scope + r_d.scope)))
    vert_p = PartitionNode(whole_scope)
    graph.add_edge(vert_p, l_d)
    graph.add_edge(vert_p, r_d)

    whole_d = RegionNode(whole_scope)
    graph.add_edge(whole_d, horiz_p)
    graph.add_edge(whole_d, vert_p)

    return whole_d


def square_from_buffer(buffer, i, j) -> List[RegionNode]:
    values = [buffer[i][j]]

    if len(buffer) > i + 1:
        values.append(buffer[i + 1][j])
    if len(buffer[i]) > j + 1:
        values.append(buffer[i][j + 1])
    if len(buffer) > i + 1 and len(buffer[i]) > j + 1:
        values.append(buffer[i + 1][j + 1])

    return values


"""
if __name__ == "__main__":

    graph = quad_tree_graph(28, 28)
    serialization.serialize(graph, "simple_image_graph.json")"""
