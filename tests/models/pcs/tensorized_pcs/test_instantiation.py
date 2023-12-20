import itertools
import math
from typing import Dict, Tuple

import torch
from torch import Tensor

from cirkit.layers.sum_product import UncollapsedCPLayer
from cirkit.models import TensorizedPC
from cirkit.models.functional import integrate
from tests import floats
from tests.models.pcs.tensorized_pcs.test_utils import get_pc_2x2_dense


def _get_pc_2x2_param_shapes() -> Dict[str, Tuple[int, ...]]:
    return {
        "input_layer.params.param": (4, 1, 1, 2),
        "inner_layers.0.params_in.param": (4, 2, 1, 1),
        "inner_layers.0.params_out.param": (4, 1, 1),
        "inner_layers.1.params_in.param": (2, 2, 1, 1),
        "inner_layers.1.params_out.param": (2, 1, 1),
        "inner_layers.2.params.param": (1, 2, 1),
    }


def _set_pc_2x2_params(pc: TensorizedPC) -> None:
    state_dict = pc.state_dict()  # type: ignore[misc]
    state_dict.update(  # type: ignore[misc]
        {  # type: ignore[misc]
            "input_layer.params.param": torch.tensor(
                # TODO: source of Any not identified
                [  # type: ignore[misc]
                    [0, 0],  # type: ignore[misc]  # 1/2, 1/2
                    [0, math.log(3)],  # type: ignore[misc]  # 1/4, 3/4
                    [0, 0],  # type: ignore[misc]  # 1/2, 1/2
                    [math.log(3), 0],  # type: ignore[misc]  # 3/4, 1/4
                ]
            ).reshape(4, 1, 1, 2),
            "inner_layers.0.params_in.param": torch.stack(
                [torch.ones(4, 1, 1) / 2, torch.ones(4, 1, 1) * 2], dim=1
            ),
            "inner_layers.0.params_out.param": torch.ones(4, 1, 1),
            "inner_layers.1.params_in.param": torch.stack(
                [torch.ones(2, 1, 1) * 2, torch.ones(2, 1, 1) / 2], dim=1
            ),
            "inner_layers.1.params_out.param": torch.ones(2, 1, 1),
            "inner_layers.2.params.param": torch.tensor(
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
    pc = get_pc_2x2_dense(layer_cls=UncollapsedCPLayer)
    param_shapes = {name: tuple(param.shape) for name, param in pc.named_parameters()}
    assert pc.num_vars == 4
    assert param_shapes == _get_pc_2x2_param_shapes()


def test_pc_output() -> None:
    pc = get_pc_2x2_dense(layer_cls=UncollapsedCPLayer)
    _set_pc_2x2_params(pc)
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(dim=-1)
    output = pc(all_inputs)
    assert output.shape == (16, 1)
    assert floats.allclose(output, _get_pc_2x2_output())


def test_pc_partition_function() -> None:
    pc = get_pc_2x2_dense(layer_cls=UncollapsedCPLayer)
    _set_pc_2x2_params(pc)
    # part_func should be 1, log is 0
    pc_pf = integrate(pc)
    assert floats.allclose(pc_pf(torch.zeros(())), 0.0)
