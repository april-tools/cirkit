import pytest

import cirkit.symbolic.functional as SF
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import PlaceholderParameter, DenseLayer, ConstantLayer, HadamardLayer
from cirkit.symbolic.params import KroneckerParameter
from cirkit.templates.region_graph import RandomBinaryTree
from tests.symbolic.test_utils import categorical_layer_factory, dense_layer_factory, \
    mixing_layer_factory, hadamard_layer_factory


def test_integrate_circuit():
    rg = RandomBinaryTree(12, depth=3, num_repetitions=2)
    sc = Circuit.from_region_graph(
        rg, num_input_units=4, num_sum_units=8,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=hadamard_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    int_sc = SF.integrate(sc)
    assert len(list(int_sc.input_layers)) == len(list(sc.input_layers))
    assert all(isinstance(isl, ConstantLayer) for isl in int_sc.input_layers)
    assert len(list(int_sc.inner_layers)) == len(list(sc.inner_layers))
    sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
    assert all(isinstance(l.weight, PlaceholderParameter) for l in sum_layers)


def test_multiply_circuits():
    rg = RandomBinaryTree(12, depth=3, num_repetitions=1)
    sc1 = Circuit.from_region_graph(
        rg, num_input_units=4, num_sum_units=8,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=hadamard_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    sc2 = Circuit.from_region_graph(
        rg, num_input_units=5, num_sum_units=3,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=hadamard_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    prod_sc = SF.multiply(sc1, sc2)
    assert len(list(prod_sc.input_layers)) == len(list(sc1.input_layers))
    assert len(list(prod_sc.input_layers)) == len(list(sc2.input_layers))
    assert len(list(prod_sc.inner_layers)) == len(list(sc1.inner_layers))
    assert len(list(prod_sc.inner_layers)) == len(list(sc2.inner_layers))
    sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), prod_sc.inner_layers))
    assert all(isinstance(l.weight, KroneckerParameter) for l in sum_layers)
    prod_layers = list(filter(lambda l: not isinstance(l, DenseLayer), prod_sc.inner_layers))
    assert all(isinstance(l, HadamardLayer) for l in prod_layers)


def test_multiply_integrate_circuits():
    rg = RandomBinaryTree(12, depth=3, num_repetitions=1)
    sc1 = Circuit.from_region_graph(
        rg, num_input_units=4, num_sum_units=8,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=hadamard_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    sc2 = Circuit.from_region_graph(
        rg, num_input_units=5, num_sum_units=3,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=hadamard_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    prod_sc = SF.multiply(sc1, sc2)
    int_sc = SF.integrate(prod_sc)
    assert len(list(int_sc.input_layers)) == len(list(prod_sc.input_layers))
    assert all(isinstance(isl, ConstantLayer) for isl in int_sc.input_layers)
    assert len(list(int_sc.inner_layers)) == len(list(prod_sc.inner_layers))
    sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
    assert all(isinstance(l.weight, PlaceholderParameter) for l in sum_layers)
