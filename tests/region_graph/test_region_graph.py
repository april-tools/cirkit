import tempfile

from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode

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
