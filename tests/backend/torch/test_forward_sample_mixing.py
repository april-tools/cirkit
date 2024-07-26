import numpy as np
import pytest
import torch
import itertools

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import MixingLayer, CategoricalLayer
from cirkit.symbolic.parameters import TensorParameter
from cirkit.symbolic.initializers import ConstantInitializer
from cirkit.pipeline import PipelineContext


def build_shallow_dense_circuit(
    folded: bool,
    n_channels: int = 3,
    n_variables: int = 11,
    n_outputs: int = 5,
    n_categories: int = 21,
) -> Circuit:
    input_layers = [
        CategoricalLayer(
            scope=range(n_variables),
            num_output_units=n_outputs,
            num_channels=n_channels,
            num_categories=n_categories,
        )
    ]

    weight = np.zeros([n_outputs, n_variables])
    for i in range(n_outputs):
        weight[i, i % n_variables] = 1

    """ 
    Will construct a deterministic weight matrix of the form:
    weight[0, 0] = 1
    ...
    weight[n_variables - 1, n_variables - 1] = 1
    weight[n_variables, 0] = 1
    ...
    """

    weight = torch.tensor(weight, dtype=torch.float32)

    dense_layer = MixingLayer(
        scope=range(n_variables),
        num_units=n_outputs,
        arity=n_variables,
        # weight=weight_parameter,
        # parameterization=lambda p: Parameter.from_unary(p, SoftmaxParameter(p.shape)),
        initializer=ConstantInitializer(weight),
    )

    input_dict = dict()
    input_dict[dense_layer] = input_layers

    output_dict = dict()
    for in_layer in input_layers:
        output_dict[in_layer] = [dense_layer]

    circuit = Circuit(
        scope=range(n_variables),
        num_channels=n_channels,
        layers=input_layers + [dense_layer],
        in_layers=input_dict,
        out_layers=output_dict,
        topologically_ordered=True,
    )
    ctx = PipelineContext(backend="torch", fold=folded, semiring="lse-sum")
    circuit = ctx.compile(circuit)
    return circuit


@pytest.mark.parametrize(
    "fold",
    itertools.product([False, True]),
)
def test_forward_sample_mixing(fold):
    n_channels = 3
    n_variables = 11
    n_outputs = 5
    n_categories = 21

    circuit = build_shallow_dense_circuit(fold, n_channels, n_variables, n_outputs, n_categories)
    samples = circuit.sample_forward(10)
    leaf_samples = samples[0]
    mixture_samples = samples[1][0]

    folds = n_variables if fold else 1

    assert leaf_samples.shape == (10, n_channels, 1, n_variables)
    assert mixture_samples.shape == (10, folds, n_dense_outputs)
