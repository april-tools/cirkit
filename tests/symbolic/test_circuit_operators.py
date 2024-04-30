import cirkit.symbolic.functional as SF
from cirkit.symbolic.layers import PlaceholderParameter, DenseLayer, ConstantLayer, HadamardLayer
from cirkit.symbolic.params import KroneckerParameter
from tests.symbolic.test_utils import build_circuit


def test_integrate_circuit():
    sc = build_circuit(12, 4, 2, num_repetitions=3)
    int_sc = SF.integrate(sc)
    assert len(list(int_sc.input_layers)) == len(list(sc.input_layers))
    assert all(isinstance(isl, ConstantLayer) for isl in int_sc.input_layers)
    assert len(list(int_sc.inner_layers)) == len(list(sc.inner_layers))
    sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
    assert all(isinstance(l.weight, PlaceholderParameter) for l in sum_layers)


def test_multiply_circuits():
    sc1 = build_circuit(12, 4, 2)
    sc2 = build_circuit(12, 3, 5)
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
    sc1 = build_circuit(12, 4, 2)
    sc2 = build_circuit(12, 3, 5)
    prod_sc = SF.multiply(sc1, sc2)
    int_sc = SF.integrate(prod_sc)
    assert len(list(int_sc.input_layers)) == len(list(prod_sc.input_layers))
    assert all(isinstance(isl, ConstantLayer) for isl in int_sc.input_layers)
    assert len(list(int_sc.inner_layers)) == len(list(prod_sc.inner_layers))
    sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
    assert all(isinstance(l.weight, PlaceholderParameter) for l in sum_layers)
