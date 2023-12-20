import tempfile

from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode


def check_region_partition_layers(rg: RegionGraph, bottom_up: bool) -> None:
    layers = rg.topological_layers(bottom_up)
    assert layers[0][0] == [], "No partition layer should exist before input"
    for partition_layer, region_layer in layers:
        for r in region_layer:
            assert isinstance(r, RegionNode), "Inconsistent layer of regions"
        for p in partition_layer:
            assert isinstance(p, PartitionNode), "Inconsistent layer of partitions"


def check_equivalent_region_graphs(rg1: RegionGraph, rg2: RegionGraph) -> None:
    rg1_nodes = sorted(rg1.nodes)
    rg2_nodes = sorted(rg2.nodes)
    assert len(rg1_nodes) == len(
        rg2_nodes
    ), f"Region graphs have not the same number of nodes: {len(rg1_nodes)} and {len(rg2_nodes)}"
    for n, m in zip(rg1_nodes, rg2_nodes):
        assert (
            n.__class__ == m.__class__
        ), "Region graphs have nodes with different types at the same locations"
        assert n.scope == m.scope, (
            f"Region graphs have nodes with different scopes at the same locations:"
            f" {n.scope} and {m.scope}"
        )


def check_region_graph_save_load(rg: RegionGraph) -> None:
    with tempfile.NamedTemporaryFile("r+") as f:
        rg.save(f.name)
        f.seek(0)
        loaded_rg = RegionGraph.load(f.name)
        check_equivalent_region_graphs(rg, loaded_rg)


def test_smoothness() -> None:
    rg = RegionGraph()
    rg.add_node(RegionNode((1, 2)))
    assert rg.is_smooth

    rg = RegionGraph()
    rg.add_node(PartitionNode((1, 2)))
    assert rg.is_smooth

    rg = RegionGraph()
    rg.add_edge(PartitionNode((1, 2)), RegionNode((1, 2)))
    assert rg.is_smooth

    rg = RegionGraph()
    rg.add_edge(PartitionNode((1, 2)), RegionNode((1, 2, 3)))
    assert not rg.is_smooth
    assert not rg.is_structured_decomposable


def test_decomposability() -> None:
    rg = RegionGraph()
    rg.add_node(RegionNode((1, 2)))
    assert rg.is_decomposable

    rg = RegionGraph()
    rg.add_node(PartitionNode((1, 2)))
    assert not rg.is_decomposable

    rg = RegionGraph()
    rg.add_edge(RegionNode((1, 2)), PartitionNode((1, 2)))
    assert rg.is_decomposable

    rg = RegionGraph()
    part = PartitionNode((1, 2))
    rg.add_edge(RegionNode((1,)), part)
    rg.add_edge(RegionNode((2,)), part)
    assert rg.is_decomposable

    rg = RegionGraph()
    part = PartitionNode((1, 2))
    rg.add_edge(RegionNode((1,)), part)
    rg.add_edge(RegionNode((1, 2)), part)
    assert not rg.is_decomposable
    assert not rg.is_structured_decomposable


def test_structured_decomposablity() -> None:
    rg = RegionGraph()
    rg.add_node(RegionNode((1, 2)))
    assert rg.is_structured_decomposable

    rg = RegionGraph()
    rg.add_node(PartitionNode((1, 2)))
    assert not rg.is_structured_decomposable

    rg = RegionGraph()
    part = PartitionNode((1, 2, 3))
    rg.add_edge(RegionNode((1, 2)), part)
    rg.add_edge(RegionNode((3,)), part)
    rg.add_edge(part, RegionNode((1, 2, 3)))
    assert rg.is_structured_decomposable

    rg = RegionGraph()
    region = RegionNode((1, 2, 3))
    part = PartitionNode((1, 2, 3))
    rg.add_edge(RegionNode((1, 2)), part)
    rg.add_edge(RegionNode((3,)), part)
    rg.add_edge(part, region)
    part = PartitionNode((1, 2, 3))
    rg.add_edge(RegionNode((1,)), part)
    rg.add_edge(RegionNode((2, 3)), part)
    rg.add_edge(part, region)
    assert not rg.is_structured_decomposable
