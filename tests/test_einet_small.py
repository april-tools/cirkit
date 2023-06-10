# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/

import itertools
import math
from typing import Dict, Tuple

import networkx as nx
import torch
from torch import Tensor

from cirkit.einet.einet import LowRankEiNet, _Args
from cirkit.einet.einsum_layer.cp_einsum_layer import CPEinsumLayer
from cirkit.einet.exp_family import CategoricalInputLayer
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode


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

    graph = nx.DiGraph()

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

    return RegionGraph(graph)


def _get_einet() -> LowRankEiNet:
    rg = _gen_rg_2x2()

    args = _Args(
        layer_type=CPEinsumLayer,
        exponential_family=CategoricalInputLayer,
        exponential_family_args={"k": 2},  # type: ignore[misc]
        num_sums=1,
        num_input=1,
        num_var=4,
        prod_exp=True,
        r=1,
    )

    einet = LowRankEiNet(rg, args)
    # TODO: we should not be required to call initialize for default init, but it builds the params
    einet.initialize(exp_reparam=False, mixing_softmax=False)
    return einet


def _get_param_shapes() -> Dict[str, Tuple[int, ...]]:
    return {
        "einet_layers.0.params": (4, 1, 1, 2),
        "einet_layers.1.cp_a": (1, 1, 4),
        "einet_layers.1.cp_b": (1, 1, 4),
        "einet_layers.1.cp_c": (1, 1, 4),
        "einet_layers.2.cp_a": (1, 1, 2),
        "einet_layers.2.cp_b": (1, 1, 2),
        "einet_layers.2.cp_c": (1, 1, 2),
        "einet_layers.3.params": (1, 1, 2),
    }


def _set_params(einet: LowRankEiNet) -> None:
    state_dict = einet.state_dict()  # type: ignore[misc]
    state_dict.update(  # type: ignore[misc]
        {  # type: ignore[misc]
            "einet_layers.0.params": torch.tensor(
                # TODO: source of Any not identified
                [  # type: ignore[misc]
                    [0, 0],  # type: ignore[misc]  # 1/2, 1/2
                    [0, math.log(3)],  # type: ignore[misc]  # 1/4, 3/4
                    [0, 0],  # type: ignore[misc]  # 1/2, 1/2
                    [math.log(3), 0],  # type: ignore[misc]  # 3/4, 1/4
                ]
            ).reshape(4, 1, 1, 2),
            "einet_layers.1.cp_a": torch.ones(1, 1, 4) / 2,
            "einet_layers.1.cp_b": torch.ones(1, 1, 4) * 2,
            "einet_layers.1.cp_c": torch.ones(1, 1, 4),
            "einet_layers.2.cp_a": torch.ones(1, 1, 2) * 2,
            "einet_layers.2.cp_b": torch.ones(1, 1, 2) / 2,
            "einet_layers.2.cp_c": torch.ones(1, 1, 2),
            "einet_layers.3.params": torch.tensor(
                [1 / 3, 2 / 3],  # type: ignore[misc]
            ).reshape(1, 1, 2),
        }
    )
    einet.load_state_dict(state_dict)  # type: ignore[misc]


def _get_output() -> Tensor:
    a = torch.tensor([1 / 2, 1 / 2]).reshape(2, 1, 1, 1)  # type: ignore[misc]
    b = torch.tensor([1 / 4, 3 / 4]).reshape(1, 2, 1, 1)  # type: ignore[misc]
    c = torch.tensor([1 / 2, 1 / 2]).reshape(1, 1, 2, 1)  # type: ignore[misc]
    d = torch.tensor([3 / 4, 1 / 4]).reshape(1, 1, 1, 2)  # type: ignore[misc]
    return torch.log((a * b * c * d)).reshape(-1, 1)


def test_einet_creation() -> None:
    einet = _get_einet()
    einet.to("cpu")
    einet.to("meta")
    param_shapes = {name: tuple(param.shape) for name, param in einet.named_parameters()}
    assert param_shapes == _get_param_shapes()


def test_einet_output() -> None:
    einet = _get_einet()
    _set_params(einet)
    possible_values = [0, 1]
    all_inputs = list(itertools.product(possible_values, repeat=4))
    output = einet(torch.tensor(all_inputs))
    assert output.shape == (16, 1)
    assert torch.allclose(output, _get_output(), rtol=0, atol=torch.finfo(torch.float32).eps)


def test_einet_partition_func() -> None:
    einet = _get_einet()
    _set_params(einet)
    # part_func should be 1, log is 0
    assert torch.allclose(einet.partition_function(), torch.zeros(()), atol=0, rtol=0)
