# pylint: disable=missing-function-docstring,missing-return-doc
# TODO: disable checking for docstrings for every test file in tests/
import itertools
import math
from typing import Dict, Tuple, Type

import torch
from torch import Tensor

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import SumProductLayer, UncollapsedCPLayer
from cirkit.models import TensorizedPC
from cirkit.models.functional import integrate
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.utils.reparams import ReparamFunction, reparam_id


def _gen_rg_2x2() -> RegionGraph:  # pylint: disable=too-many-locals
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


def get_pc_from_region_graph(
    rg: RegionGraph,
    num_units: int = 1,
    layer_cls: Type[SumProductLayer] = UncollapsedCPLayer,
    reparam: ReparamFunction = reparam_id,
) -> TensorizedPC:
    layer_kwargs = {"rank": 1} if layer_cls == UncollapsedCPLayer else {}  # type: ignore[misc]
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


def _get_pc_2x2() -> TensorizedPC:
    rg = _gen_rg_2x2()
    return get_pc_from_region_graph(rg)


def _get_param_shapes() -> Dict[str, Tuple[int, ...]]:
    return {
        "input_layer.params": (4, 1, 1, 2),
        "inner_layers.0.params_in": (4, 2, 1, 1),
        "inner_layers.0.params_out": (4, 1, 1),
        "inner_layers.1.params_in": (2, 2, 1, 1),
        "inner_layers.1.params_out": (2, 1, 1),
        "inner_layers.2.params": (1, 2, 1),
    }


def _set_params(pc: TensorizedPC) -> None:
    state_dict = pc.state_dict()  # type: ignore[misc]
    state_dict.update(  # type: ignore[misc]
        {  # type: ignore[misc]
            "input_layer.params": torch.tensor(
                # TODO: source of Any not identified
                [  # type: ignore[misc]
                    [0, 0],  # type: ignore[misc]  # 1/2, 1/2
                    [0, math.log(3)],  # type: ignore[misc]  # 1/4, 3/4
                    [0, 0],  # type: ignore[misc]  # 1/2, 1/2
                    [math.log(3), 0],  # type: ignore[misc]  # 3/4, 1/4
                ]
            ).reshape(4, 1, 1, 2),
            "inner_layers.0.params_in": torch.stack(
                [torch.ones(4, 1, 1) / 2, torch.ones(4, 1, 1) * 2], dim=1
            ),
            "inner_layers.0.params_out": torch.ones(4, 1, 1),
            "inner_layers.1.params_in": torch.stack(
                [torch.ones(2, 1, 1) * 2, torch.ones(2, 1, 1) / 2], dim=1
            ),
            "inner_layers.1.params_out": torch.ones(2, 1, 1),
            "inner_layers.2.params": torch.tensor(
                [1 / 3, 2 / 3],  # type: ignore[misc]
            ).reshape(1, 2, 1),
        }
    )
    pc.load_state_dict(state_dict)  # type: ignore[misc]


def _get_output() -> Tensor:
    a = torch.tensor([1 / 2, 1 / 2]).reshape(2, 1, 1, 1)  # type: ignore[misc]
    b = torch.tensor([1 / 4, 3 / 4]).reshape(1, 2, 1, 1)  # type: ignore[misc]
    c = torch.tensor([1 / 2, 1 / 2]).reshape(1, 1, 2, 1)  # type: ignore[misc]
    d = torch.tensor([3 / 4, 1 / 4]).reshape(1, 1, 1, 2)  # type: ignore[misc]
    return torch.log((a * b * c * d)).reshape(-1, 1)


def test_pc_instantiation() -> None:
    pc = _get_pc_2x2()
    param_shapes = {name: tuple(param.shape) for name, param in pc.named_parameters()}
    assert pc.num_variables == 4
    assert param_shapes == _get_param_shapes()


def test_pc_output() -> None:
    pc = _get_pc_2x2()
    _set_params(pc)
    all_inputs = list(itertools.product([0, 1], repeat=4))
    output = pc(torch.tensor(all_inputs))
    assert output.shape == (16, 1)
    assert torch.allclose(output, _get_output(), rtol=0, atol=torch.finfo(torch.float32).eps)


def test_pc_partition_function() -> None:
    pc = _get_pc_2x2()
    _set_params(pc)
    # part_func should be 1, log is 0
    pc_pf = integrate(pc)
    assert torch.allclose(pc_pf(), torch.zeros(()), atol=0, rtol=0)
