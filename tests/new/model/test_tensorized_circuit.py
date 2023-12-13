# pylint: disable=missing-function-docstring

import itertools
from typing import Dict, Tuple

import torch
from torch import Tensor

from cirkit.new import set_layer_comp_space
from cirkit.new.layers import CategoricalLayer, CPLayer
from cirkit.new.model import TensorizedCircuit
from cirkit.new.region_graph import QuadTree
from cirkit.new.reparams import LeafReparam, LogSoftmaxReparam
from cirkit.new.symbolic import SymbolicTensorizedCircuit
from tests import floats


def _get_circuit_2x2() -> TensorizedCircuit:
    rg = QuadTree((2, 2), struct_decomp=False)
    symbc = SymbolicTensorizedCircuit(
        rg,
        num_input_units=1,
        num_sum_units=1,
        input_layer_cls=CategoricalLayer,
        input_layer_kwargs={"num_categories": 2},  # type: ignore[misc]
        input_reparam=LogSoftmaxReparam,
        sum_layer_cls=CPLayer,
        sum_layer_kwargs={},  # type: ignore[misc]
        sum_reparam=LeafReparam,
        prod_layer_cls=CPLayer,
        prod_layer_kwargs={},  # type: ignore[misc]
    )
    return TensorizedCircuit(symbc, num_channels=1)


def _get_circuit_2x2_param_shapes() -> Dict[str, Tuple[int, ...]]:
    return {
        "layers.0.params.reparams.0.param": (1, 1, 1, 2),  # Input for {0}.
        "layers.1.params.param": (1, 1),  # Dense after above.
        # TODO: should we have two Dense for {0,1} and {0,2}?
        "layers.2.params.reparams.0.param": (1, 1, 1, 2),  # Input for {1}.
        "layers.3.params.param": (1, 1),  # Dense after above.
        "layers.4.params.reparams.0.param": (1, 1, 1, 2),  # Input for {2}.
        "layers.5.params.param": (1, 1),  # Dense after above.
        "layers.6.params.reparams.0.param": (1, 1, 1, 2),  # Input for {3}.
        "layers.7.params.param": (1, 1),  # Dense after above.
        "layers.8.sum.params.param": (1, 1),  # CP of {0, 1}.
        "layers.9.sum.params.param": (1, 1),  # CP of {0, 2}.
        "layers.10.sum.params.param": (1, 1),  # CP of {1, 3}.
        "layers.11.sum.params.param": (1, 1),  # CP of {2, 3}.
        "layers.12.sum.params.param": (1, 1),  # CP of {0, 1 + 2, 3}.
        "layers.13.sum.params.param": (1, 1),  # CP of {0, 2 + 1, 3}.
        "layers.14.params.param": (1, 2),  # Mixing of {0, 1, 2, 3}.
    }


def _set_circuit_2x2_params(circuit: TensorizedCircuit) -> None:
    state_dict = circuit.state_dict()  # type: ignore[misc]
    state_dict.update(  # type: ignore[misc]
        {  # type: ignore[misc]
            "layers.0.params.reparams.0.param": (  # Input for {0}.
                torch.tensor([1 / 2, 1 / 2]).log().view(1, 1, 1, 2)  # type: ignore[misc]
            ),
            "layers.1.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after above.
            "layers.2.params.reparams.0.param": (  # Input for {1}.
                torch.tensor([1 / 4, 3 / 4]).log().view(1, 1, 1, 2)  # type: ignore[misc]
            ),
            "layers.3.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after above.
            "layers.4.params.reparams.0.param": (  # Input for {2}.
                torch.tensor([1 / 2, 1 / 2]).log().view(1, 1, 1, 2)  # type: ignore[misc]
            ),
            "layers.5.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after above.
            "layers.6.params.reparams.0.param": (  # Input for {3}.
                torch.tensor([3 / 4, 1 / 4]).log().view(1, 1, 1, 2)  # type: ignore[misc]
            ),
            "layers.7.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after above.
            "layers.8.sum.params.param": torch.tensor(2 / 1).view(1, 1),  # CP of {0, 1}.
            "layers.9.sum.params.param": torch.tensor(2 / 1).view(1, 1),  # CP of {0, 2}.
            "layers.10.sum.params.param": torch.tensor(1 / 2).view(1, 1),  # CP of {1, 3}.
            "layers.11.sum.params.param": torch.tensor(1 / 2).view(1, 1),  # CP of {2, 3}.
            "layers.12.sum.params.param": torch.tensor(1 / 1).view(1, 1),  # CP of {0, 1 + 2, 3}.
            "layers.13.sum.params.param": torch.tensor(1 / 1).view(1, 1),  # CP of {0, 2 + 1, 3}.
            "layers.14.params.param": (  # Mixing of {0, 1, 2, 3}.
                torch.tensor([1 / 3, 2 / 3]).view(1, 2)  # type: ignore[misc]
            ),
        }
    )
    circuit.load_state_dict(state_dict)  # type: ignore[misc]


def _get_circuit_2x2_output() -> Tensor:
    a = torch.tensor([1 / 2, 1 / 2]).reshape(2, 1, 1, 1)  # type: ignore[misc]
    b = torch.tensor([1 / 4, 3 / 4]).reshape(1, 2, 1, 1)  # type: ignore[misc]
    c = torch.tensor([1 / 2, 1 / 2]).reshape(1, 1, 2, 1)  # type: ignore[misc]
    d = torch.tensor([3 / 4, 1 / 4]).reshape(1, 1, 1, 2)  # type: ignore[misc]
    return (a * b * c * d).reshape(-1, 1, 1)


def test_circuit_instantiation() -> None:
    circuit = _get_circuit_2x2()
    param_shapes = {name: tuple(param.shape) for name, param in circuit.named_parameters()}
    assert circuit.num_vars == 4
    assert param_shapes == _get_circuit_2x2_param_shapes()


def test_circuit_output_linear() -> None:
    set_layer_comp_space("linear")
    circuit = _get_circuit_2x2()
    _set_circuit_2x2_params(circuit)
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(
        dim=-1
    )  # shape (B=16, D=2, C=1).
    output = circuit(all_inputs)
    assert output.shape == (16, 1, 1)  # shape (B=16, num_out=1, num_cls=1)
    # TODO: this is currently not correct. how to fix???
    assert floats.allclose(output, _get_circuit_2x2_output())
    set_layer_comp_space("log")  # TODO: use a with to tmp set default?


def test_circuit_output_log() -> None:
    circuit = _get_circuit_2x2()
    _set_circuit_2x2_params(circuit)
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(
        dim=-1
    )  # shape (B=16, D=2, C=1).
    output = circuit(all_inputs)
    assert output.shape == (16, 1, 1)  # shape (B=16, num_out=1, num_cls=1)
    assert floats.allclose(output, _get_circuit_2x2_output().log())


def test_circuit_part_func() -> None:
    circuit = _get_circuit_2x2()
    _set_circuit_2x2_params(circuit)
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(
        dim=-1
    )  # shape (B=16, D=2, C=1).
    output = circuit(all_inputs)  # shape (B=16, num_out=1, num_cls=1)
    sum_output = torch.logsumexp(output, dim=0)
    part_func = circuit.partition_func
    assert floats.allclose(part_func, sum_output)
    assert floats.allclose(part_func, 0.0)
