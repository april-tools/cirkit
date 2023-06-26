# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/

import itertools
import math
from typing import Callable, Dict, List, Tuple, Union

import pytest
import torch
from torch import Tensor

from cirkit.layers.einsum.cp import CPLayer
from cirkit.layers.exp_family import CategoricalLayer
from cirkit.models.einet import LowRankEiNet
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree
from cirkit.utils import RandomCtx


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


def _get_einet() -> LowRankEiNet:
    rg = _gen_rg_2x2()

    einet = LowRankEiNet(
        rg,
        layer_type=CPLayer,  # type: ignore[misc]
        exponential_family=CategoricalLayer,
        exponential_family_args={"k": 2},  # type: ignore[misc]
        num_sums=1,
        num_input=1,
        num_var=4,
        prod_exp=True,
        r=1,
    )
    # TODO: we should not be required to call initialize for default init, but it builds the params
    einet.initialize(exp_reparam=False, mixing_softmax=False)
    return einet


def _get_param_shapes() -> Dict[str, Tuple[int, ...]]:
    return {
        "einet_layers.0.params": (4, 1, 1, 2),
        "einet_layers.1.param_left": (1, 1, 4),
        "einet_layers.1.param_right": (1, 1, 4),
        "einet_layers.1.param_out": (1, 1, 4),
        "einet_layers.2.param_left": (1, 1, 2),
        "einet_layers.2.param_right": (1, 1, 2),
        "einet_layers.2.param_out": (1, 1, 2),
        "einet_layers.3.param": (1, 1, 2),
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
            "einet_layers.1.param_left": torch.ones(1, 1, 4) / 2,
            "einet_layers.1.param_right": torch.ones(1, 1, 4) * 2,
            "einet_layers.1.param_out": torch.ones(1, 1, 4),
            "einet_layers.2.param_left": torch.ones(1, 1, 2) * 2,
            "einet_layers.2.param_right": torch.ones(1, 1, 2) / 2,
            "einet_layers.2.param_out": torch.ones(1, 1, 2),
            "einet_layers.3.param": torch.tensor(
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


@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs,log_answer",
    [
        (PoonDomingos, {"shape": [4, 4], "delta": 2}, 10.94407844543457),
        (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}, 52.791404724121094),
        (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}, 23.796138763427734),
    ],
)
@RandomCtx(42)
def test_einet_partition_function(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    log_answer: float,
) -> None:
    """Tests the creation and partition of an einet.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
        log_answer (float): The answer of partition func. NOTE: we don't know if it's correct, but \
            it guarantees reproducibility.
    """
    # TODO: type of kwargs should be refined
    device = "cpu"

    graph = rg_cls(**kwargs)

    einet = LowRankEiNet(
        graph,
        layer_type=CPLayer,  # type: ignore[misc]
        exponential_family=CategoricalLayer,
        exponential_family_args={"k": 2},  # type: ignore[misc]
        num_sums=16,
        num_input=16,
        num_var=16,
        prod_exp=True,
        r=1,
    )
    einet.initialize(exp_reparam=False, mixing_softmax=False)
    einet.to(device)

    # Generate all possible combinations of 16 integers from the list of possible values
    possible_values = [0, 1]
    all_lists = list(itertools.product(possible_values, repeat=16))

    # compute outputs
    out = einet(torch.tensor(all_lists))

    # log sum exp on outputs to compute their sum
    # TODO: for simple log-sum-exp, pytorch have implementation
    sum_out = torch.logsumexp(out, dim=0, keepdim=True)

    assert torch.isclose(sum_out, torch.tensor(log_answer), rtol=1e-6, atol=0)
    assert torch.isclose(einet.partition_function(), sum_out, rtol=1e-6, atol=0)
