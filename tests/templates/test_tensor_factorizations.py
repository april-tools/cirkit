import itertools

import pytest

from cirkit.symbolic.layers import EmbeddingLayer, HadamardLayer, SumLayer
from cirkit.symbolic.parameters import ConstantParameter
from cirkit.templates import tensor_factorizations
from cirkit.templates.utils import Parameterization
from cirkit.utils.scope import Scope


@pytest.mark.parametrize(
    "rank,param,weighted",
    itertools.product(
        [1, 5],
        [None, Parameterization(activation="softmax", initialization="normal")],
        [False, True],
    ),
)
def test_cp(rank: int, param: Parameterization | None, weighted: bool):
    shape = (256, 32, 32)
    circuit = tensor_factorizations.cp(shape, rank, param=param, weighted=weighted)
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
    if param is None:
        assert all(len(sl.weight.nodes) == 1 for sl in input_layers)
    else:
        assert all(len(sl.weight.nodes) == 2 for sl in input_layers)
    if weighted:
        if param is None:
            assert len(sum_layers[0].weight.nodes) == 1
        else:
            assert len(sum_layers[0].weight.nodes) == 2
    else:
        assert len(sum_layers[0].weight.nodes) == 1
        assert isinstance(sum_layers[0].weight.nodes[0], ConstantParameter)
