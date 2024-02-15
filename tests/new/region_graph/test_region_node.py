from cirkit.new.region_graph import PartitionNode, RegionGraph, RegionNode


def test_smoothness() -> None:
    rg = RegionGraph()
    rg.add_node(RegionNode((1, 2)))
    rg.freeze()

    assert rg.is_smooth 


def test_decomposability() -> None:
    rg = RegionGraph()
    rg.add_node(RegionNode((1, 2)))
    rg.freeze()
    assert rg.is_decomposable 


def test_structured_decomposablity() -> None:
    rg = RegionGraph()
    rg.add_node(RegionNode((1, 2)))
    rg.freeze()
    assert rg.is_structured_decomposable 


def test_structure_decomposability2() -> None:
    rg = RegionGraph()
    part = PartitionNode((1, 2, 3))
    rg.add_edge(RegionNode((1, 2)), part)
    rg.add_edge(RegionNode((3,)), part)
    rg.add_edge(part, RegionNode((1, 2, 3)))
    rg.freeze()
    assert rg.is_structured_decomposable 
