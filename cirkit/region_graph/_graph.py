from itertools import count
from typing import List

import networkx as nx
import numpy as np


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

def quad_tree_graph(width: int, height: int, stdec=False) -> nx.DiGraph:

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
            hypercube = ((i, j), (i+1, j+1))

            c_scope = hypercube_to_scope(hypercube, shape)
            c_node = RegionNode(c_scope)
            graph.add_node(c_node)
            buffer[i].append(c_node)

    lr_choice = 0 # random.choice([0, 1])
    td_choice = 0 # random.choice([0, 1])

    old_buffer_width = width
    old_buffer_height = height
    old_buffer = buffer

    while old_buffer_width != 1 and old_buffer_height != 1:
        buffer_width = old_buffer_width // 2 if old_buffer_width % 2 == 0 else old_buffer_width // 2 + 1
        buffer_height = old_buffer_height // 2 if old_buffer_height % 2 == 0 else old_buffer_height // 2 + 1
        buffer = [[] for _ in range(buffer_width)]

        # lr_idx = [2 * i + lr_choice for i in range(old_buffer_width // 2 + 1)]
        # td_idx = [2 * j + td_choice for j in range(old_buffer_height // 2 + 1)]

        for i in range(buffer_width):
            buffer[i] = [[] for _ in range(buffer_height)]
            for j in range(buffer_height):
                regions = square_from_buffer(old_buffer, 2 * i + lr_choice, 2 * j + td_choice)

                if len(regions) == 1:
                    buffer[i][j] = regions[0]
                elif len(regions) == 2:
                    buffer[i][j] = merge_2_regions(regions, graph)
                elif len(regions) == 4:
                    buffer[i][j] = merge_4_regions(regions, graph, stdec)
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


def merge_4_regions(regions: List[RegionNode], graph: nx.DiGraph, stdec: bool) -> RegionNode:

    assert len(regions) ==4

    if not stdec:
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

        # MERGE L & R
        assert whole_scope == list(set(sorted(l_d.scope + r_d.scope)))
        vert_p = PartitionNode(whole_scope)
        graph.add_edge(vert_p, l_d)
        graph.add_edge(vert_p, r_d)

        # Mix
        whole_d = RegionNode(whole_scope)
        graph.add_edge(whole_d, horiz_p)
        graph.add_edge(whole_d, vert_p)

        return whole_d

    else: # Horizontal and then vertical
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

        whole_d = RegionNode(whole_scope)
        graph.add_edge(whole_d, horiz_p)

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


def topological_layers_bottom_up(graph):
    """
    Arranging the PC graph in topological layers -- see Algorithm 1 in the paper.

    :param graph: the PC graph (DiGraph)
    :return: list of layers, alternating between DistributionVector and Product layers (list of lists of nodes).
    """
    sums = list(sorted(get_sums(graph)))
    products = list(sorted(get_products(graph)))
    leaves = list(sorted(get_leaves(graph)))

    visited_nodes = set(leaves)
    layers = []

    num_nodes = len(leaves) + len(sums) + len(products)

    while len(visited_nodes) != num_nodes:
        product_layer = [p for p in products if p not in visited_nodes and all([s in visited_nodes for s in graph.successors(p)])]
        product_layer = sorted(product_layer)
        layers.append(product_layer)
        visited_nodes.update(product_layer)

        sum_layer = [s for s in sums if s not in visited_nodes and all([p in visited_nodes for p in graph.successors(s)])]
        sum_layer = sorted(sum_layer)
        layers.append(sum_layer)
        visited_nodes.update(sum_layer)

    layers.insert(0, leaves)

    return layers

"""
if __name__ == "__main__":

    graph = quad_tree_graph(28, 28)
    serialization.serialize(graph, "simple_image_graph.json")"""
