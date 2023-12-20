import pytest

from cirkit.region_graph import PartitionNode, RegionNode


def test_region_node() -> None:
    node = RegionNode((1, 2))
    assert repr(node) == "RegionNode: {1, 2}"

    with pytest.raises(AssertionError, match="The scope of a node must be non-empty"):
        node = RegionNode(())


def test_partition_node() -> None:
    node = PartitionNode((1, 2))
    assert repr(node) == "PartitionNode: {1, 2}"

    with pytest.raises(AssertionError, match="The scope of a node must be non-empty"):
        node = PartitionNode(())


def test_compare() -> None:
    region1 = RegionNode((1, 2))
    region2 = RegionNode((1, 3))

    partition1 = PartitionNode((1, 2))
    partition2 = PartitionNode((1, 3))

    assert region1 < region2 < partition1 < partition2
