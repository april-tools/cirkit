from typing import Type

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import SumProductLayer, UncollapsedCPLayer
from cirkit.models import TensorizedPC
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory


def _gen_rg_2x2_dense() -> RegionGraph:  # pylint: disable=too-many-locals
    reg0 = RegionNode({0})
    reg1 = RegionNode({1})
    reg2 = RegionNode({2})
    reg3 = RegionNode({3})

    part01 = PartitionNode({0, 1})
    part23 = PartitionNode({2, 3})
    part02 = PartitionNode({0, 2})
    part13 = PartitionNode({1, 3})

    reg01 = RegionNode({0, 1})
    reg23 = RegionNode({2, 3})
    reg02 = RegionNode({0, 2})
    reg13 = RegionNode({1, 3})

    part01_23 = PartitionNode({0, 1, 2, 3})
    part02_13 = PartitionNode({0, 1, 2, 3})

    reg0123 = RegionNode({0, 1, 2, 3})

    graph = RegionGraph()

    graph.add_edge(reg0, part01)
    graph.add_edge(reg0, part02)
    graph.add_edge(reg1, part01)
    graph.add_edge(reg1, part13)
    graph.add_edge(reg2, part02)
    graph.add_edge(reg2, part23)
    graph.add_edge(reg3, part13)
    graph.add_edge(reg3, part23)

    graph.add_edge(part01, reg01)
    graph.add_edge(part23, reg23)
    graph.add_edge(part02, reg02)
    graph.add_edge(part13, reg13)

    graph.add_edge(reg01, part01_23)
    graph.add_edge(reg23, part01_23)
    graph.add_edge(reg02, part02_13)
    graph.add_edge(reg13, part02_13)

    graph.add_edge(part01_23, reg0123)
    graph.add_edge(part02_13, reg0123)

    return graph


def _gen_rg_5_sparse() -> RegionGraph:  # pylint: disable=too-many-locals,too-many-statements
    reg0 = RegionNode({0})
    reg1 = RegionNode({1})
    reg2 = RegionNode({2})
    reg3 = RegionNode({3})
    reg4 = RegionNode({4})

    graph = RegionGraph()

    part01 = PartitionNode({0, 1})
    part23 = PartitionNode({2, 3})
    part123a = PartitionNode({1, 2, 3})
    part123b = PartitionNode({1, 2, 3})

    reg01 = RegionNode({0, 1})
    reg23 = RegionNode({2, 3})
    reg123a = RegionNode({1, 2, 3})
    reg123b = RegionNode({1, 2, 3})

    graph.add_edge(reg0, part01)
    graph.add_edge(reg1, part01)
    graph.add_edge(reg2, part23)
    graph.add_edge(reg3, part23)
    graph.add_edge(reg1, part123a)
    graph.add_edge(reg2, part123a)
    graph.add_edge(reg3, part123a)
    graph.add_edge(reg1, part123b)
    graph.add_edge(reg2, part123b)
    graph.add_edge(reg3, part123b)

    graph.add_edge(part01, reg01)
    graph.add_edge(part23, reg23)
    graph.add_edge(part123a, reg123a)
    graph.add_edge(part123b, reg123b)

    part0_123a = PartitionNode({0, 1, 2, 3})
    part0_123b = PartitionNode({0, 1, 2, 3})
    reg0_123 = RegionNode({0, 1, 2, 3})

    graph.add_edge(reg0, part0_123a)
    graph.add_edge(reg123a, part0_123a)
    graph.add_edge(reg0, part0_123b)
    graph.add_edge(reg123b, part0_123b)
    graph.add_edge(part0_123a, reg0_123)
    graph.add_edge(part0_123b, reg0_123)

    part01_23a = PartitionNode({0, 1, 2, 3})
    part01_23b = PartitionNode({0, 1, 2, 3})
    part01_23c = PartitionNode({0, 1, 2, 3})
    reg01_23 = RegionNode({0, 1, 2, 3})

    graph.add_edge(reg01, part01_23a)
    graph.add_edge(reg23, part01_23a)
    graph.add_edge(reg01, part01_23b)
    graph.add_edge(reg23, part01_23b)
    graph.add_edge(reg01, part01_23c)
    graph.add_edge(reg23, part01_23c)
    graph.add_edge(part01_23a, reg01_23)
    graph.add_edge(part01_23b, reg01_23)
    graph.add_edge(part01_23c, reg01_23)

    part01234a = PartitionNode({0, 1, 2, 3, 4})
    part01234b = PartitionNode({0, 1, 2, 3, 4})
    reg01234 = RegionNode({0, 1, 2, 3, 4})

    graph.add_edge(reg4, part01234a)
    graph.add_edge(reg0_123, part01234a)
    graph.add_edge(reg4, part01234b)
    graph.add_edge(reg01_23, part01234b)
    graph.add_edge(part01234a, reg01234)
    graph.add_edge(part01234b, reg01234)

    return graph


def get_pc_from_region_graph(
    rg: RegionGraph,
    num_units: int = 1,
    layer_cls: Type[SumProductLayer] = UncollapsedCPLayer,
    reparam: ReparamFactory = ReparamIdentity,
) -> TensorizedPC:
    layer_kwargs = {"rank": 1} if layer_cls == UncollapsedCPLayer else {}
    pc = TensorizedPC.from_region_graph(
        rg,
        layer_cls=layer_cls,
        efamily_cls=CategoricalLayer,
        layer_kwargs=layer_kwargs,
        efamily_kwargs={"num_categories": 2},  # type: ignore[misc]
        num_inner_units=num_units,
        num_input_units=num_units,
        reparam=reparam,
    )
    return pc


def get_pc_2x2_dense(
    reparam: ReparamFactory = ReparamIdentity,
    layer_cls: Type[SumProductLayer] = UncollapsedCPLayer,
    num_units: int = 1,
) -> TensorizedPC:
    rg = _gen_rg_2x2_dense()
    return get_pc_from_region_graph(rg, num_units=num_units, layer_cls=layer_cls, reparam=reparam)


def get_pc_5_sparse(
    reparam: ReparamFactory = ReparamIdentity,
    layer_cls: Type[SumProductLayer] = UncollapsedCPLayer,
    num_units: int = 1,
) -> TensorizedPC:
    rg = _gen_rg_5_sparse()
    return get_pc_from_region_graph(rg, num_units=num_units, layer_cls=layer_cls, reparam=reparam)


def test_rg_2x2_sparse() -> None:
    rg = _gen_rg_2x2_dense()
    assert rg.is_smooth
    assert rg.is_decomposable
    assert not rg.is_structured_decomposable
    assert len(list(rg.output_nodes)) == 1
    assert len(list(rg.input_nodes)) == 4


def test_rg_5_sparse() -> None:
    rg = _gen_rg_5_sparse()
    assert rg.is_smooth
    assert rg.is_decomposable
    assert not rg.is_structured_decomposable
    assert len(list(rg.output_nodes)) == 1
    assert len(list(rg.input_nodes)) == 5
