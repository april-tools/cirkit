# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/
import functools
import itertools
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import pytest
import torch
from torch import Tensor

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import CPLayer
from cirkit.models.functional import integrate
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree
from cirkit.utils import RandomCtx
from cirkit.utils.reparams import (
    ReparamFunction,
    reparam_exp,
    reparam_id,
    reparam_positive,
    reparam_softmax,
    reparam_square,
)


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


def _get_pc_2x2() -> TensorizedPC:
    rg = _gen_rg_2x2()

    pc = TensorizedPC.from_region_graph(
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


def test_small_pc_partition_function() -> None:
    pc = _get_pc_2x2()
    _set_params(pc)
    # part_func should be 1, log is 0
    pc_pf = integrate(pc)
    assert torch.allclose(pc_pf(), torch.zeros(()), atol=0, rtol=0)


def _get_deep_pc(  # type: ignore[misc]
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    reparam_func: ReparamFunction = reparam_id,
) -> TensorizedPC:
    # TODO: type of kwargs should be refined
    rg = rg_cls(**kwargs)
    pc = TensorizedPC.from_region_graph(
        rg,
        layer_cls=CPLayer,  # type: ignore[misc]
        efamily_cls=CategoricalLayer,
        layer_kwargs={"rank": 1, "prod_exp": True},  # type: ignore[misc]
        efamily_kwargs={"num_categories": 2},  # type: ignore[misc]
        num_inner_units=16,
        num_input_units=16,
        reparam=reparam_func,
    )
    return pc


@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs,true_log_z",
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
def test_pc_marginalization(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    true_log_z: Optional[float],
) -> None:
    """Tests the creation and variable marginalization on a PC.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
        true_log_z (Optional[float]): The answer of partition func.
            NOTE: we don't know if it's correct, but it guarantees reproducibility.
    """
    pc = _get_deep_pc(rg_cls, kwargs)  # type: ignore[misc]
    num_vars = pc.num_variables

    # Generate all possible combinations of 16 integers from the list of possible values
    possible_values = [0, 1]
    all_data = torch.tensor(
        list(itertools.product(possible_values, repeat=num_vars))  # type: ignore[misc]
    )

    # Instantiate the integral of the PC, i.e., computing the partition function
    pc_pf = integrate(pc)
    log_z = pc_pf()
    assert log_z.shape == (1, 1)

    # Compute outputs
    log_scores = pc(all_data)
    lls = log_scores - log_z

    # Check the partition function computation
    assert torch.isclose(log_z, torch.logsumexp(log_scores, dim=0, keepdim=True), rtol=1e-6, atol=0)

    # Compare the partition function against the answer, if given
    if true_log_z is not None:
        assert torch.isclose(log_z, torch.tensor(true_log_z), rtol=1e-6, atol=0), f"{log_z.item()}"

    # Perform variable marginalization on the last two variables
    mar_data = all_data[::4]
    mar_scores = pc.integrate(mar_data, [num_vars - 2, num_vars - 1])
    mar_lls = mar_scores - log_z

    # Check the results of marginalization
    sum_lls = torch.logsumexp(lls.view(-1, 4), dim=1, keepdim=True)
    assert mar_lls.shape[0] == lls.shape[0] // 4 and len(mar_lls.shape) == len(lls.shape)
    assert torch.allclose(sum_lls, mar_lls, rtol=1e-6, atol=torch.finfo(torch.float32).eps)


@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs,reparam_name",
    [
        (rg_cls, kwargs, reparam_name)
        for (rg_cls, kwargs) in (
            (PoonDomingos, {"shape": [4, 4], "delta": 2}),
            (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}),
            (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}),
            (PoonDomingos, {"shape": [3, 3], "delta": 2}),
            (QuadTree, {"width": 3, "height": 3, "struct_decomp": False}),
            (QuadTree, {"width": 3, "height": 3, "struct_decomp": True}),
            (RandomBinaryTree, {"num_vars": 9, "depth": 3, "num_repetitions": 2}),
        )
        for reparam_name in ("exp", "square", "softmax", "positive")
    ],
)
@RandomCtx(42)
def test_einet_nonneg_reparams(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    reparam_name: str,
) -> None:
    """Tests multiple non-negative re-parametrizations on tensorized circuits.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
        reparam_name (str): The reparametrization function identifier.
    """
    if reparam_name == "exp":
        reparam_func = reparam_exp
    elif reparam_name == "square":
        reparam_func = reparam_square
    elif reparam_name == "softmax":
        reparam_func = functools.partial(reparam_softmax, dim=-2)
    elif reparam_name == "positive":
        reparam_func = functools.partial(reparam_positive, eps=1e-7)
    else:
        assert False

    pc = _get_deep_pc(rg_cls, kwargs, reparam_func=reparam_func)  # type: ignore[misc]
    num_vars = pc.num_variables

    # Generate all possible combinations of 16 integers from the list of possible values
    possible_values = [0, 1]
    all_data = torch.tensor(
        list(itertools.product(possible_values, repeat=num_vars))  # type: ignore[misc]
    )

    # Instantiate the integral of the PC, i.e., computing the partition function
    pc_pf = integrate(pc)
    log_z = pc_pf()
    assert log_z.shape == (1, 1)

    # Compute outputs
    log_scores = pc(all_data)

    # Check the partition function computation
    assert torch.isclose(log_z, torch.logsumexp(log_scores, dim=0, keepdim=True), atol=5e-7)

    # The circuit should be already normalized,
    #  if the re-parameterization is via softmax and using normalized input distributions
    if reparam_name == "softmax":
        assert torch.allclose(log_z, torch.zeros(()), atol=5e-7)
