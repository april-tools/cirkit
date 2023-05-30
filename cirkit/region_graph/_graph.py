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
