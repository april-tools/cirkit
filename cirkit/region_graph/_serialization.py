import json
from typing import Dict, List

import networkx as nx

from cirkit.region_graph._graph import (
    PartitionNode,
    RegionNode,
    get_leaves,
    get_products,
    get_roots,
    poon_domingos_structure,
    quad_tree_graph,
)


def deserialize(path: str) -> nx.DiGraph:
    """
    Deserializes a graph.

    :param path:
    :return:
    """
    graph_json = json.load(open(path, "r"))
    regions: Dict[str, List[int, ...]] = graph_json["regions"]
    region_nodes: Dict[str, RegionNode] = {idx: RegionNode(scope) for idx, scope in regions.items()}

    graph = nx.DiGraph()

    for partition in graph_json["graph"]:
        parent_node = region_nodes[str(partition["p"])]
        left_child_node = region_nodes[str(partition["l"])]
        right_child_node = region_nodes[str(partition["r"])]

        product_node = PartitionNode(parent_node.scope)

        graph.add_edge(parent_node, product_node)
        graph.add_edge(product_node, left_child_node)
        graph.add_edge(product_node, right_child_node)

    for node in get_leaves(graph):
        node.einet_address.replica_idx = 0

    return graph


def serialize(graph: nx.DiGraph, path: str):
    graph_json: Dict = {}

    root: RegionNode = get_roots(graph)[0]

    # insert "regions"
    regions: List[RegionNode] = get_region_nodes(graph)
    regions_dict: Dict[RegionNode, int] = {node: n for n, node in enumerate(regions)}
    graph_json["regions"] = {str(n): list(map(int, node.scope)) for node, n in regions_dict.items()}

    products = get_products(graph)

    partitions = []
    for product in products:
        children = list(graph.successors(product))
        parent = list(graph.predecessors(product))
        assert len(children) == 2
        assert len(parent) == 1

        partitions.append(
            {
                "p": regions_dict[parent[0]],
                "l": regions_dict[children[0]],
                "r": regions_dict[children[1]],
            }
        )

    graph_json["graph"] = partitions
    print(graph_json)
    json.dump(graph_json, open(path, "w"))


def get_region_nodes(graph):
    return [n for n in graph.nodes() if type(n) == RegionNode]


if __name__ == "__main__":
    structure = "quad_tree_dec"

    if structure == "quad_tree":
        graph = quad_tree_graph(width=28, height=28, stdec=False)
        serialize(graph, "quad_tree_28_28.json")
    elif structure == "quad_tree_dec":
        graph = quad_tree_graph(width=28, height=28, stdec=True)
        serialize(graph, "quad_tree_stdec_28_28.json")
    elif structure == "poon_domingos":
        pd_num_pieces = [4]
        pd_delta = [[28 / d, 28 / d] for d in pd_num_pieces]
        graph = poon_domingos_structure((28, 28), pd_delta)
        serialize(graph, "poon_domingos_28_28.json")
    else:
        raise AssertionError("Unknown region graph")
