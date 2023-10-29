# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/
import itertools
import math
from typing import Dict, Tuple

import torch
from torch import Tensor

from cirkit.layers.sum_product import UncollapsedCPLayer
from cirkit.models import TensorizedPC
from cirkit.models.functional import integrate
from tests.models.pcs.tensorized_pcs.test_utils import get_pc_2x2_dense


def _get_pc_2x2_param_shapes() -> Dict[str, Tuple[int, ...]]:
    return {
        "input_layer.params": (4, 1, 1, 2),
        "inner_layers.0.params_in": (4, 2, 1, 1),
        "inner_layers.0.params_out": (4, 1, 1),
        "inner_layers.1.params_in": (2, 2, 1, 1),
        "inner_layers.1.params_out": (2, 1, 1),
        "inner_layers.2.params": (1, 2, 1),
    }


def _set_pc_2x2_params(pc: TensorizedPC) -> None:
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


def _get_pc_2x2_output() -> Tensor:
    a = torch.tensor([1 / 2, 1 / 2]).reshape(2, 1, 1, 1)  # type: ignore[misc]
    b = torch.tensor([1 / 4, 3 / 4]).reshape(1, 2, 1, 1)  # type: ignore[misc]
    c = torch.tensor([1 / 2, 1 / 2]).reshape(1, 1, 2, 1)  # type: ignore[misc]
    d = torch.tensor([3 / 4, 1 / 4]).reshape(1, 1, 1, 2)  # type: ignore[misc]
    return torch.log((a * b * c * d)).reshape(-1, 1)


def test_pc_instantiation() -> None:
    pc = get_pc_2x2_dense(layer_cls=UncollapsedCPLayer)  # type: ignore[misc]
    param_shapes = {name: tuple(param.shape) for name, param in pc.named_parameters()}
    assert pc.num_variables == 4
    assert param_shapes == _get_pc_2x2_param_shapes()


def test_pc_output() -> None:
    pc = get_pc_2x2_dense(layer_cls=UncollapsedCPLayer)  # type: ignore[misc]
    _set_pc_2x2_params(pc)
    all_inputs = list(itertools.product([0, 1], repeat=4))
    output = pc(torch.tensor(all_inputs))
    assert output.shape == (16, 1)
    assert torch.allclose(output, _get_pc_2x2_output(), rtol=0, atol=torch.finfo(torch.float32).eps)


def test_pc_partition_function() -> None:
    pc = get_pc_2x2_dense(layer_cls=UncollapsedCPLayer)  # type: ignore[misc]
    _set_pc_2x2_params(pc)
    # part_func should be 1, log is 0
    pc_pf = integrate(pc)
    assert torch.allclose(pc_pf(), torch.zeros(()), atol=0, rtol=0)
