# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/

import itertools
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import pytest
import torch
from torch import Tensor

from cirkit.atlas.integrate import IntegrationContext
from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import CPLayer
from cirkit.models.tensorized_circuit import TensorizedPC
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


def _get_pc() -> TensorizedPC:
    rg = _gen_rg_2x2()

    pc = TensorizedPC(
        rg,
        layer_cls=CPLayer,  # type: ignore[misc]
        efamily_cls=CategoricalLayer,
        layer_kwargs={"rank": 1, "prod_exp": True},  # type: ignore[misc]
        efamily_kwargs={"num_categories": 2},  # type: ignore[misc]
        num_inner_units=1,
        num_input_units=1,
    )
    return pc


def _get_param_shapes() -> Dict[str, Tuple[int, ...]]:
    return {
        "input_layer.params": (4, 1, 1, 2),
        "inner_layers.0.params_left": (4, 1, 1),
        "inner_layers.0.params_right": (4, 1, 1),
        "inner_layers.0.params_out": (4, 1, 1),
        "inner_layers.1.params_left": (2, 1, 1),
        "inner_layers.1.params_right": (2, 1, 1),
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
            "inner_layers.0.params_left": torch.ones(4, 1, 1) / 2,
            "inner_layers.0.params_right": torch.ones(4, 1, 1) * 2,
            "inner_layers.0.params_out": torch.ones(4, 1, 1),
            "inner_layers.1.params_left": torch.ones(2, 1, 1) * 2,
            "inner_layers.1.params_right": torch.ones(2, 1, 1) / 2,
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


def test_pc_creation() -> None:
    pc = _get_pc()
    # TODO: what to test here?
    param_shapes = {name: tuple(param.shape) for name, param in pc.named_parameters()}
    assert param_shapes == _get_param_shapes()


def test_pc_output() -> None:
    pc = _get_pc()
    _set_params(pc)
    possible_values = [0, 1]
    all_inputs = list(itertools.product(possible_values, repeat=4))
    output = pc(torch.tensor(all_inputs))
    assert output.shape == (16, 1)
    assert torch.allclose(output, _get_output(), rtol=0, atol=torch.finfo(torch.float32).eps)


def test_small_pc_partition_function() -> None:
    pc = _get_pc()
    _set_params(pc)
    # part_func should be 1, log is 0
    pc_pf = pc.integrate()
    assert torch.allclose(pc_pf(torch.zeros(())), torch.zeros(()), atol=0, rtol=0)


def _get_deep_pc(  # type: ignore[misc]
    rg_cls: Callable[..., RegionGraph], kwargs: Dict[str, Union[int, bool, List[int]]]
) -> TensorizedPC:
    # TODO: type of kwargs should be refined
    rg = rg_cls(**kwargs)
    pc = TensorizedPC(
        rg,
        layer_cls=CPLayer,  # type: ignore[misc]
        efamily_cls=CategoricalLayer,
        layer_kwargs={"rank": 1, "prod_exp": True},  # type: ignore[misc]
        efamily_kwargs={"num_categories": 2},  # type: ignore[misc]
        num_inner_units=16,
        num_input_units=16,
    )
    return pc


@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs,log_answer",
    [
        (PoonDomingos, {"shape": [4, 4], "delta": 2}, 10.246478080749512),
        (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}, 51.94971466064453),
        (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}, 24.000484466552734),
        (PoonDomingos, {"shape": [3, 3], "delta": 2}, None),
        (QuadTree, {"width": 3, "height": 3, "struct_decomp": False}, None),
        (QuadTree, {"width": 3, "height": 3, "struct_decomp": True}, None),
        (RandomBinaryTree, {"num_vars": 9, "depth": 3, "num_repetitions": 2}, None),
    ],
)
@RandomCtx(42)
def test_pc_partition_function(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    log_answer: Optional[float],
) -> None:
    """Tests the creation and partition function computation on a PC.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
        log_answer (Optional[float]): The answer of partition func.
            NOTE: we don't know if it's correct, but it guarantees reproducibility.
    """
    pc = _get_deep_pc(rg_cls, kwargs)  # type: ignore[misc]

    # Generate all possible combinations of 16 integers from the list of possible values
    possible_values = [0, 1]
    all_lists = list(itertools.product(possible_values, repeat=pc.num_variables))

    # compute outputs
    out = pc(torch.tensor(all_lists))

    # log sum exp on outputs to compute their sum
    # TODO: for simple log-sum-exp, pytorch have implementation
    sum_out = torch.logsumexp(out, dim=0, keepdim=True)

    pc_pf = pc.integrate()
    assert torch.isclose(pc_pf(torch.zeros(())), sum_out, rtol=1e-6, atol=0)
    if log_answer is not None:
        assert torch.isclose(
            sum_out, torch.tensor(log_answer), rtol=1e-6, atol=0
        ), f"{sum_out.item()}"


@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs",
    [
        (PoonDomingos, {"shape": [4, 4], "delta": 2}),
        (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}),
        (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}),
        (PoonDomingos, {"shape": [3, 3], "delta": 2}),
        (QuadTree, {"width": 3, "height": 3, "struct_decomp": False}),
        (QuadTree, {"width": 3, "height": 3, "struct_decomp": True}),
        (RandomBinaryTree, {"num_vars": 9, "depth": 3, "num_repetitions": 2}),
    ],
)
@RandomCtx(42)
def test_pc_marginalization(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
) -> None:
    """Tests variables marginalization on a PC.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
    """
    pc = _get_deep_pc(rg_cls, kwargs)  # type: ignore[misc]
    num_vars = pc.num_variables

    # Compute log-likelihoods
    all_data = torch.tensor(list(itertools.product([0, 1], repeat=num_vars)))  # type: ignore[misc]
    lls = pc(all_data)

    # Perform marginalization on the last two variables
    mar_pc = pc.integrate(icontext=IntegrationContext(num_vars, ivars=[num_vars - 2, num_vars - 1]))
    mar_data = all_data[::4]
    mar_lls = mar_pc(mar_data)

    # Check the results of marginalization
    sum_lls = torch.logsumexp(lls.view(-1, 4), dim=1, keepdim=True)
    assert mar_lls.shape[0] == lls.shape[0] // 4 and len(mar_lls.shape) == len(lls.shape)
    # TODO: when working in log-space it is REALLY difficult to get atol < 5e-7 in my experience.
    #  We might decide to run tests using float64 arithmetic, because we only care about correctness
    #  regardless of arithmetic precision.
    assert torch.allclose(sum_lls, mar_lls, rtol=1e-6, atol=5e-7)
