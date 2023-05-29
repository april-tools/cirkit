import json
from typing import Dict, List, TypedDict, Union

import torch
from pyjuice.graph.region_graph import InnerRegionNode, InputRegionNode, PartitionNode, RegionGraph
from pyjuice.layer.input_layers.categorical_layer import CategoricalLayer


class _RGJson(TypedDict):
    """The structure of region graph json file."""

    regions: Dict[str, List[int]]
    graph: List[Dict[str, int]]


def _dfs_build(
    key: int,
    nodes_cache: Dict[int, Union[InnerRegionNode, InputRegionNode]],
    graph_json: _RGJson,
    num_latents: int,
    is_root: bool,
) -> Union[InnerRegionNode, InputRegionNode]:
    """Build the region graph recursively.

    Args:
        key (int): The current region to build.
        nodes_cache (Dict[int, Union[InnerRegionNode, InputRegionNode]]): The cache for built nodes.
        graph_json (_RGJson): The graph structure loaded from json.
        num_latents (int): Num latents for region graph nodes.
        is_root (bool): Whether `key` is the global root.

    Returns:
        Union[InnerRegionNode, InputRegionNode]: Region graph rooted at `key`.
    """
    if key in nodes_cache:
        return nodes_cache[key]

    if len(scope := graph_json["regions"][str(key)]) == 1:  # we only accept uni-var input layer
        nodes_cache[key] = InputRegionNode(scope, num_latents, CategoricalLayer, num_cats=256)
        return nodes_cache[key]

    partitions: List[PartitionNode] = []
    for partition in graph_json["graph"]:
        if partition["p"] == key:
            left = _dfs_build(partition["l"], nodes_cache, graph_json, num_latents, False)
            right = _dfs_build(partition["r"], nodes_cache, graph_json, num_latents, False)

            edge_ids = torch.arange(num_latents).view(-1, 1).repeat(1, 2)
            partitions.append(PartitionNode([left, right], num_latents, edge_ids))
    assert len(partitions) == 1  # we don't know what to do with more than 1

    num_out = 1 if is_root else num_latents
    parent_ids, child_ids = torch.meshgrid(
        torch.arange(num_out), torch.arange(num_latents), indexing="ij"
    )
    nodes_cache[key] = InnerRegionNode(
        partitions, num_out, torch.stack((parent_ids, child_ids), dim=0).view(2, -1)
    )
    return nodes_cache[key]


def load_region_graph(filename: str, num_latents: int) -> RegionGraph:
    """Load region graph from json file.

    Args:
        filename (str): The filename for region graph.
        num_latents (int): Num latents to build the graph.

    Returns:
        RegionGraph: The root node of loaded region graph.
    """
    with open(filename, "r", encoding="utf-8") as f:
        graph_json: _RGJson = json.load(f)

    all_children = set(partition["l"] for partition in graph_json["graph"]).union(
        partition["r"] for partition in graph_json["graph"]
    )
    all_regions = set(int(key) for key in graph_json["regions"].keys())
    assert all_children.issubset(all_regions)
    root_region = all_regions.difference(all_children)
    assert len(root_region) == 1

    return _dfs_build(root_region.pop(), {}, graph_json, num_latents, True)
