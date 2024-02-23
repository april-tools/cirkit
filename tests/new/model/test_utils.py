from typing import Any, Dict, Literal, Tuple, Type

import torch
from torch import Tensor

from cirkit.new.layers import (
    CategoricalLayer,
    CPLayer,
    InputLayer,
    NormalLayer,
    ProductLayer,
    SumLayer,
)
from cirkit.new.model import TensorizedCircuit
from cirkit.new.region_graph import QuadTree
from cirkit.new.reparams import EFNormalReparam, LeafReparam, LogSoftmaxReparam
from cirkit.new.symbolic import SymbolicTensorizedCircuit
from cirkit.new.utils.type_aliases import ReparamFactory, SymbCfgFactory


# pylint: disable-next=too-many-arguments,dangerous-default-value
def get_circuit_2x2_fullcfg(  # type: ignore[misc]
    *,
    num_channels: int = 1,
    num_input_units: int = 1,
    num_sum_units: int = 1,
    num_classes: int = 1,
    input_layer_cls: Type[InputLayer] = CategoricalLayer,
    input_layer_kwargs: Dict[str, Any] = {"num_categories": 2},
    input_reparam: ReparamFactory = LogSoftmaxReparam,
    sum_layer_cls: Type[SumLayer] = CPLayer,
    sum_layer_kwargs: Dict[str, Any] = {},
    sum_reparam: ReparamFactory = LeafReparam,
    prod_layer_cls: Type[ProductLayer] = CPLayer,
    prod_layer_kwargs: Dict[str, Any] = {},
) -> TensorizedCircuit:
    rg = QuadTree((2, 2), struct_decomp=False)
    symbc = SymbolicTensorizedCircuit(
        rg,
        num_channels=num_channels,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=num_classes,
        input_cfg=SymbCfgFactory(
            layer_cls=input_layer_cls,
            layer_kwargs=input_layer_kwargs,  # type: ignore[misc]
            reparam_factory=input_reparam,
        ),
        sum_cfg=SymbCfgFactory(
            layer_cls=sum_layer_cls,
            layer_kwargs=sum_layer_kwargs,  # type: ignore[misc]
            reparam_factory=sum_reparam,
        ),
        prod_cfg=SymbCfgFactory(
            layer_cls=prod_layer_cls, layer_kwargs=prod_layer_kwargs  # type: ignore[misc]
        ),
    )
    return TensorizedCircuit(symbc)


def get_circuit_2x2(setting: Literal["cat", "norm"] = "cat") -> TensorizedCircuit:
    if setting == "cat":
        return get_circuit_2x2_fullcfg()
    if setting == "norm":
        return get_circuit_2x2_fullcfg(
            input_layer_cls=NormalLayer,
            input_layer_kwargs={},  # type: ignore[misc]
            input_reparam=EFNormalReparam,
        )
    assert False, "This should not happen."


def get_circuit_2x2_param_shapes(
    setting: Literal["cat", "norm"] = "cat"
) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]]
    if setting == "cat":
        shapes = {
            "layers.0.params.reparams.0.param": (1, 1, 1, 2),  # Input for {0}.
            "layers.2.params.reparams.0.param": (1, 1, 1, 2),  # Input for {1}.
            "layers.4.params.reparams.0.param": (1, 1, 1, 2),  # Input for {2}.
            "layers.6.params.reparams.0.param": (1, 1, 1, 2),  # Input for {3}.
        }
    elif setting == "norm":
        shapes = {
            "layers.0.params.reparams.0.param": (1, 1, 2, 1),  # Input for {0}.
            "layers.2.params.reparams.0.param": (1, 1, 2, 1),  # Input for {1}.
            "layers.4.params.reparams.0.param": (1, 1, 2, 1),  # Input for {2}.
            "layers.6.params.reparams.0.param": (1, 1, 2, 1),  # Input for {3}.
        }
    else:
        assert False, "This should not happen."

    shapes.update(
        {
            # TODO: should we have two Dense for {0,1} and {0,2}?
            "layers.1.params.param": (1, 1),  # Dense after input {0}.
            "layers.3.params.param": (1, 1),  # Dense after input {1}.
            "layers.5.params.param": (1, 1),  # Dense after input {2}.
            "layers.7.params.param": (1, 1),  # Dense after input {3}.
            "layers.8.sum_layer.params.param": (1, 1),  # CP of {0, 1}.
            "layers.9.sum_layer.params.param": (1, 1),  # CP of {0, 2}.
            "layers.10.sum_layer.params.param": (1, 1),  # CP of {1, 3}.
            "layers.11.sum_layer.params.param": (1, 1),  # CP of {2, 3}.
            "layers.12.sum_layer.params.param": (1, 1),  # CP of {{0, 1}, {2, 3}}.
            "layers.13.sum_layer.params.param": (1, 1),  # CP of {{0, 2}, {1, 3}}.
            "layers.14.params.param": (1, 2),  # Mixing of {0, 1, 2, 3}.
        }
    )
    return shapes


def set_circuit_2x2_params(
    circuit: TensorizedCircuit, setting: Literal["cat", "norm"] = "cat"
) -> None:
    state_dict = circuit.state_dict()  # type: ignore[misc]
    if setting == "cat":
        state_dict.update(  # type: ignore[misc]
            {  # type: ignore[misc]
                "layers.0.params.reparams.0.param": (  # Input for {0}.
                    torch.tensor([1 / 2, 1 / 2]).log().view(1, 1, 1, 2)  # type: ignore[misc]
                ),
                "layers.2.params.reparams.0.param": (  # Input for {1}.
                    torch.tensor([1 / 4, 3 / 4]).log().view(1, 1, 1, 2)  # type: ignore[misc]
                ),
                "layers.4.params.reparams.0.param": (  # Input for {2}.
                    torch.tensor([1 / 2, 1 / 2]).log().view(1, 1, 1, 2)  # type: ignore[misc]
                ),
                "layers.6.params.reparams.0.param": (  # Input for {3}.
                    torch.tensor([3 / 4, 1 / 4]).log().view(1, 1, 1, 2)  # type: ignore[misc]
                ),
            }
        )
    elif setting == "norm":
        state_dict.update(  # type: ignore[misc]
            {  # type: ignore[misc]
                "layers.0.params.reparams.0.param": (  # Input for {0}.
                    torch.tensor([0.0, 0.0]).view(1, 1, 2, 1)  # type: ignore[misc]
                ),
                "layers.2.params.reparams.0.param": (  # Input for {1}.
                    torch.tensor([0.0, 0.0]).view(1, 1, 2, 1)  # type: ignore[misc]
                ),
                "layers.4.params.reparams.0.param": (  # Input for {2}.
                    torch.tensor([0.0, 0.0]).view(1, 1, 2, 1)  # type: ignore[misc]
                ),
                "layers.6.params.reparams.0.param": (  # Input for {3}.
                    torch.tensor([0.0, 0.0]).view(1, 1, 2, 1)  # type: ignore[misc]
                ),
            }
        )
    else:
        assert False, "This should not happen."

    state_dict.update(  # type: ignore[misc]
        {  # type: ignore[misc]
            "layers.1.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after input {0}.
            "layers.3.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after input {1}.
            "layers.5.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after input {2}.
            "layers.7.params.param": torch.tensor(1 / 1).view(1, 1),  # Dense after input {3}.
            "layers.8.sum_layer.params.param": torch.tensor(2 / 1).view(1, 1),  # CP of {0, 1}.
            "layers.9.sum_layer.params.param": torch.tensor(2 / 1).view(1, 1),  # CP of {0, 2}.
            "layers.10.sum_layer.params.param": torch.tensor(1 / 2).view(1, 1),  # CP of {1, 3}.
            "layers.11.sum_layer.params.param": torch.tensor(1 / 2).view(1, 1),  # CP of {2, 3}.
            "layers.12.sum_layer.params.param": (  # CP of {{0, 1}, {2, 3}}.
                torch.tensor(1 / 1).view(1, 1)
            ),
            "layers.13.sum_layer.params.param": (  # CP of {{0, 2}, {1, 3}}.
                torch.tensor(1 / 1).view(1, 1)
            ),
            "layers.14.params.param": (  # Mixing of {0, 1, 2, 3}.
                torch.tensor([1 / 3, 2 / 3]).view(1, 2)  # type: ignore[misc]
            ),
        }
    )
    circuit.load_state_dict(state_dict)  # type: ignore[misc]


def get_circuit_2x2_output(setting: Literal["cat", "norm"] = "cat") -> Tensor:
    if setting == "cat":
        a = torch.tensor([1 / 2, 1 / 2]).reshape(2, 1, 1, 1)  # type: ignore[misc]
        b = torch.tensor([1 / 4, 3 / 4]).reshape(1, 2, 1, 1)  # type: ignore[misc]
        c = torch.tensor([1 / 2, 1 / 2]).reshape(1, 1, 2, 1)  # type: ignore[misc]
        d = torch.tensor([3 / 4, 1 / 4]).reshape(1, 1, 1, 2)  # type: ignore[misc]
        return (a * b * c * d).reshape(-1, 1, 1)
    raise NotImplementedError
