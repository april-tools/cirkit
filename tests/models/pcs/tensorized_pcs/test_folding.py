# pylint: disable=missing-function-docstring
import functools
import itertools

import pytest
import torch

from cirkit.models import TensorizedPC
from cirkit.models.functional import integrate
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.utils.reparams import reparam_exp, reparam_softmax
from tests.models.pcs.tensorized_pcs.test_instantiation import get_pc_from_region_graph


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


def test_rg_5_sparse() -> None:
    rg = _gen_rg_5_sparse()
    assert rg.is_smooth
    assert rg.is_decomposable
    assert not rg.is_structured_decomposable
    assert len(list(rg.output_nodes)) == 1
    assert len(list(rg.input_nodes)) == 5


def _get_pc_5_sparse(normalized: bool) -> TensorizedPC:
    rg = _gen_rg_5_sparse()
    reparam = functools.partial(reparam_softmax, dim=-2) if normalized else reparam_exp
    return get_pc_from_region_graph(rg, num_units=2, reparam=reparam)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "normalized",
    [(False,), (True,)],
)
def test_pc_sparse(normalized: bool) -> None:
    pc = _get_pc_5_sparse(normalized)
    assert any(should_pad for (should_pad, _, __) in pc.bookkeeping)
    assert any(not should_pad for (should_pad, _, _) in pc.bookkeeping)
    data = torch.tensor(list(itertools.product([0, 1], repeat=5)))  # type: ignore[misc]
    pc_pf = integrate(pc)
    log_z = pc_pf()
    log_scores = pc(data)
    lls = log_scores - log_z
    assert torch.allclose(torch.logsumexp(lls, dim=0), torch.zeros(()), atol=3e-7)
    if normalized:
        assert torch.allclose(log_z, torch.zeros(()), atol=3e-7)


@torch.set_grad_enabled(True)
@pytest.mark.parametrize(
    "normalized",
    [(False,), (True,)],
)
def test_pc_sparse_backprop(normalized: bool) -> None:
    pc = _get_pc_5_sparse(normalized)
    opt = torch.optim.SGD(pc.parameters(), lr=0.01)
    data = torch.tensor(list(itertools.product([0, 1], repeat=5)))  # type: ignore[misc]
    pc_pf = integrate(pc)
    log_z = pc_pf()
    log_scores = pc(data)
    lls = log_scores - log_z
    loss = -torch.mean(lls)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    opt.zero_grad()
    log_z = pc_pf()
    log_scores = pc(data)
    lls = log_scores - log_z
    new_loss = -torch.mean(lls)
    assert torch.isfinite(new_loss)
    assert new_loss < loss
