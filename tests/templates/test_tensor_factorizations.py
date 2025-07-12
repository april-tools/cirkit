import itertools
from typing import cast

import numpy as np
import pytest

from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.symbolic.dtypes import DataType
from cirkit.symbolic.layers import (
    CategoricalLayer,
    EmbeddingLayer,
    HadamardLayer,
    KroneckerLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import ConstantParameter, SoftmaxParameter, TensorParameter
from cirkit.templates import tensor_factorizations
from cirkit.templates.utils import Parameterization
from cirkit.utils.scope import Scope


@pytest.mark.parametrize("rank", [1, 5])
def test_factorization_cp(rank: int) -> None:
    shape = (48, 16, 32)
    circuit = tensor_factorizations.cp(shape, rank, input_layer="embedding")
    assert circuit.scope == Scope(range(len(shape)))
    input_layers = list(circuit.inputs)
    product_layers = list(circuit.product_layers)
    sum_layers = list(circuit.sum_layers)
    assert len(input_layers) == len(shape)
    assert all(isinstance(sl, EmbeddingLayer) for sl in input_layers)
    assert len(product_layers) == 1 and isinstance(product_layers[0], HadamardLayer)
    assert product_layers[0].num_input_units == rank
    assert product_layers[0].arity == len(input_layers)
    assert len(sum_layers) == 1 and isinstance(sum_layers[0], SumLayer)
    assert sum_layers[0].num_input_units == rank and sum_layers[0].num_output_units == 1
    assert sum_layers[0].arity == 1
    assert all(len(cast(EmbeddingLayer, sl).weight.nodes) == 1 for sl in input_layers)
    assert len(sum_layers[0].weight.nodes) == 1
    assert isinstance(sum_layers[0].weight.nodes[0], ConstantParameter)


@pytest.mark.parametrize("rank", [1, 5])
def test_factorization_cp_probabilistic(rank: int) -> None:
    shape = (48, 16, 32)
    circuit = tensor_factorizations.cp(
        shape,
        rank,
        input_layer="categorical",
        input_params={"probs": Parameterization(activation="softmax")},
        weight_param=Parameterization(activation="softmax"),
    )
    assert circuit.scope == Scope(range(len(shape)))
    input_layers = list(circuit.inputs)
    product_layers = list(circuit.product_layers)
    sum_layers = list(circuit.sum_layers)
    assert len(input_layers) == len(shape)
    assert all(isinstance(sl, CategoricalLayer) for sl in input_layers)
    assert len(product_layers) == 1 and isinstance(product_layers[0], HadamardLayer)
    assert product_layers[0].num_input_units == rank
    assert product_layers[0].arity == len(input_layers)
    assert len(sum_layers) == 1 and isinstance(sum_layers[0], SumLayer)
    assert sum_layers[0].num_input_units == rank and sum_layers[0].num_output_units == 1
    assert sum_layers[0].arity == 1
    assert all(len(cast(CategoricalLayer, sl).probs.nodes) == 2 for sl in input_layers)
    assert all(SoftmaxParameter in map(type, sl.probs.nodes) for sl in input_layers)
    assert len(sum_layers[0].weight.nodes) == 2
    assert SoftmaxParameter in map(type, sum_layers[0].weight.nodes)


@pytest.mark.parametrize("rank", [1, 5])
def test_factorization_tucker(rank: int):
    shape = (48, 16, 32)
    circuit = tensor_factorizations.tucker(shape, rank, input_layer="embedding")
    assert circuit.scope == Scope(range(len(shape)))
    input_layers = list(circuit.inputs)
    product_layers = list(circuit.product_layers)
    sum_layers = list(circuit.sum_layers)
    assert len(input_layers) == len(shape)
    assert all(isinstance(sl, EmbeddingLayer) for sl in input_layers)
    assert len(product_layers) == 1 and isinstance(product_layers[0], KroneckerLayer)
    assert product_layers[0].num_input_units == rank
    assert product_layers[0].arity == len(input_layers)
    assert len(sum_layers) == 1 and isinstance(sum_layers[0], SumLayer)
    assert sum_layers[0].num_input_units == rank ** len(input_layers)
    assert sum_layers[0].num_output_units == 1
    assert sum_layers[0].arity == 1
    assert all(len(sl.weight.nodes) == 1 for sl in input_layers)
    assert len(sum_layers[0].weight.nodes) == 1
    assert isinstance(sum_layers[0].weight.nodes[0], TensorParameter)


@pytest.mark.parametrize("rank", [1, 5])
def test_factorization_tucker_probabilistic(rank: int):
    shape = (48, 16, 32)
    circuit = tensor_factorizations.tucker(
        shape,
        rank,
        input_layer="categorical",
        input_params={"probs": Parameterization(activation="softmax")},
        core_param=Parameterization(activation="softmax"),
    )
    assert circuit.scope == Scope(range(len(shape)))
    input_layers = list(circuit.inputs)
    product_layers = list(circuit.product_layers)
    sum_layers = list(circuit.sum_layers)
    assert len(input_layers) == len(shape)
    assert all(isinstance(sl, CategoricalLayer) for sl in input_layers)
    assert len(product_layers) == 1 and isinstance(product_layers[0], KroneckerLayer)
    assert product_layers[0].num_input_units == rank
    assert product_layers[0].arity == len(input_layers)
    assert len(sum_layers) == 1 and isinstance(sum_layers[0], SumLayer)
    assert sum_layers[0].num_input_units == rank ** len(input_layers)
    assert sum_layers[0].num_output_units == 1
    assert sum_layers[0].arity == 1
    assert all(len(cast(CategoricalLayer, sl).probs.nodes) == 2 for sl in input_layers)
    assert all(SoftmaxParameter in map(type, sl.probs.nodes) for sl in input_layers)
    assert len(sum_layers[0].weight.nodes) == 2
    assert SoftmaxParameter in map(type, sum_layers[0].weight.nodes)


@pytest.mark.parametrize(
    "rank,factor_param",
    itertools.product([1, 5], [None, Parameterization(dtype="complex")]),
)
def test_factorization_tensor_train(rank: int, factor_param: Parameterization | None):
    shape = (48, 16, 32)
    circuit = tensor_factorizations.tensor_train(shape, rank=rank, factor_param=factor_param)
    assert circuit.scope == Scope(range(len(shape)))
    input_layers = list(circuit.inputs)
    product_layers = list(circuit.product_layers)
    sum_layers = list(circuit.sum_layers)
    assert set([sl.num_output_units for sl in input_layers]) == {rank}
    assert len([sl for sl in input_layers if circuit.layer_scope(sl) == Scope([0])]) == 1
    assert (
        len([sl for sl in input_layers if circuit.layer_scope(sl) == Scope([len(shape) - 1])]) == 1
    )
    for i in range(1, len(shape) - 1, 1):
        assert len([sl for sl in input_layers if circuit.layer_scope(sl) == Scope([i])]) == rank
    assert all(isinstance(sl, HadamardLayer) for sl in product_layers)
    assert len(product_layers) == (len(shape) - 2) * rank + 1
    assert len(sum_layers) == len(shape) - 1
    for sl in sum_layers:
        assert len(sl.weight.nodes) == 1
        weight = sl.weight.nodes[0]
        assert isinstance(weight, ConstantParameter)
        value = np.reshape(weight.value, shape=(sl.num_output_units, sl.arity, sl.num_input_units))
        ones = np.ones(sl.num_input_units)
        zeros = np.zeros(sl.num_input_units)
        for i in range(sl.num_output_units):
            assert np.all(value[i, i] == ones)
            for j in range(sl.num_output_units):
                if i == j:
                    continue
                assert np.all(value[i, j] == zeros)
    if factor_param is not None:
        for sl in input_layers:
            assert len(sl.weight.nodes) == 1
            node_tensor = cast(TorchTensorParameter, sl.weight.nodes[0])
            assert node_tensor.dtype == DataType.COMPLEX
