
from cirkit.new.region_graph import PartitionNode, RegionNode


def test_region_node() -> None:
    node = RegionNode((1, 2))
    node_id = hex(id(node))
    assert repr(node) == "RegionNode@" + node_id + "(Scope({1, 2}))"


def test_partition_node() -> None:
    node = PartitionNode((1, 2))
    node_id = hex(id(node))
    assert repr(node) == "PartitionNode@" + node_id + "(Scope({1, 2}))"


def test_compare1() -> None:
    region1 = RegionNode((1, 2))
    region2 = RegionNode((1, 4))

    assert region1 < region2


def test_compare2a() -> None:
    region1 = RegionNode((1, 2))
    partition1 = PartitionNode((1, 2))

    assert partition1 < region1


def test_compare2b() -> None:
    region1 = RegionNode((1, 5))
    partition1 = PartitionNode((1, 5))

    assert partition1 < region1


def test_compare3() -> None:
    region1 = RegionNode((1, 5))
    partition1 = PartitionNode((1, 2))

    assert partition1 < region1


def test_compare4() -> None:
    region1 = RegionNode((1, 2))
    partition1 = PartitionNode((1, 5))

    assert region1 < partition1
