import numpy as np
import pytest
import torch
import itertools

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import DenseLayer, CategoricalLayer
from cirkit.symbolic.parameters import TensorParameter, Parameter
from cirkit.symbolic.initializers import ConstantInitializer
from cirkit.pipeline import PipelineContext
from cirkit.utils.scope import Scope


def build_shallow_dense_circuit(
    folded: bool,
    n_channels: int = 3,
    n_variables: int = 11,
    n_dense_inputs: int = 8,
    n_dense_outputs: int = 5,
    n_categories: int = 21,
) -> TorchCircuit:
    input_layers = [
        CategoricalLayer(
            scope=Scope(range(n_variables)),
            num_output_units=n_dense_inputs,
            num_channels=n_channels,
            num_categories=n_categories,
        )
    ]

    weight = np.zeros([n_dense_inputs, n_dense_outputs])
    for i in range(n_dense_inputs):
        weight[i, i % n_dense_outputs] = 1
    weight = torch.tensor(weight, dtype=torch.float32)

    """ 
    Will construct a deterministic weight matrix of the form:
    weight[0, 0] = 1
    ...
    weight[n_dense_outputs - 1, n_dense_outputs - 1] = 1
    weight[n_dense_outputs, 0] = 1
    ...
    """

    dense_layer = DenseLayer(
        scope=Scope(range(n_variables)),
        num_input_units=n_dense_inputs,
        num_output_units=n_dense_outputs,
        # weight=weight_parameter,
        weight_factory=lambda shape: Parameter.from_leaf(TensorParameter(
            *shape, initializer=ConstantInitializer(weight), learnable=False
        ))
    )

    input_dict = dict()
    input_dict[dense_layer] = input_layers

    circuit = Circuit(
        scope=Scope(range(n_variables)),
        num_channels=n_channels,
        layers=input_layers + [dense_layer],
        outputs=[dense_layer],
        in_layers=input_dict,
        topologically_ordered=True
    )
    ctx = PipelineContext(backend="torch", fold=folded, semiring="lse-sum")
    circuit = ctx.compile(circuit)
    return circuit


@pytest.mark.parametrize(
    "fold",
    itertools.product([False, True]),
)
def test_forward_sample_dense(fold):
    n_channels = 3
    n_variables = 11
    n_dense_inputs = 8
    n_dense_outputs = 5
    n_categories = 21

    circuit = build_shallow_dense_circuit(
        fold, n_channels, n_variables, n_dense_inputs, n_dense_outputs, n_categories
    )
    samples = circuit.sample_forward(10)
    leaf_samples = samples[0]
    mixture_samples = samples[1][0]

    folds = n_variables if fold else 1

    assert leaf_samples.shape == (10, n_channels, n_dense_outputs, n_variables)
    assert mixture_samples.shape == (10, folds, n_dense_outputs)
