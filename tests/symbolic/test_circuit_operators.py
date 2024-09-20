import itertools

import pytest

import cirkit.symbolic.functional as SF
from cirkit.symbolic.circuit import are_compatible
from cirkit.symbolic.layers import DenseLayer, HadamardLayer, LogPartitionLayer
from cirkit.symbolic.parameters import ConjugateParameter, KroneckerParameter, ReferenceParameter
from tests.symbolic.test_utils import build_multivariate_monotonic_structured_cpt_pc


@pytest.mark.parametrize(
    "num_units,input_layer",
    itertools.product([1, 3], ["bernoulli", "gaussian"]),
)
def test_integrate_circuit(num_units: int, input_layer: str):
    sc = build_multivariate_monotonic_structured_cpt_pc(
        num_units=num_units, input_layer=input_layer
    )
    int_sc = SF.integrate(sc)
    assert int_sc.is_smooth
    assert int_sc.is_decomposable
    assert int_sc.is_structured_decomposable
    assert not int_sc.is_omni_compatible
    assert not int_sc.scope
    assert len(list(int_sc.inputs)) == len(list(sc.inputs))
    assert all(isinstance(isl, LogPartitionLayer) for isl in int_sc.inputs)
    assert len(list(int_sc.inner_layers)) == len(list(sc.inner_layers))
    dense_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
    assert all(isinstance(r, ReferenceParameter) for l in dense_layers for r in l.weight.inputs)


@pytest.mark.parametrize(
    "num_units,input_layer",
    itertools.product([1, 3], ["bernoulli", "gaussian"]),
)
def test_multiply_circuits(num_units: int, input_layer: str):
    sc1 = build_multivariate_monotonic_structured_cpt_pc(
        num_units=num_units, input_layer=input_layer
    )
    sc2 = build_multivariate_monotonic_structured_cpt_pc(
        num_units=num_units * 2 + 1, input_layer=input_layer
    )
    prod_num_units = num_units * (num_units * 2 + 1)
    sc = SF.multiply(sc1, sc2)
    assert are_compatible(sc1, sc) and are_compatible(sc, sc1)
    assert are_compatible(sc2, sc) and are_compatible(sc, sc2)
    assert len(list(sc.inputs)) == len(list(sc1.inputs))
    assert len(list(sc.inputs)) == len(list(sc2.inputs))
    assert len(list(sc.inner_layers)) == len(list(sc1.inner_layers))
    assert len(list(sc.inner_layers)) == len(list(sc2.inner_layers))
    assert all(l.num_output_units == prod_num_units for l in sc.inputs)
    dense_layers = list(filter(lambda l: isinstance(l, DenseLayer), sc.inner_layers))
    assert all(isinstance(l.weight.output, KroneckerParameter) for l in dense_layers)
    assert all(
        l.weight.shape == (prod_num_units, prod_num_units)
        for l in dense_layers
        if sc.layer_outputs(l)
    )
    prod_layers = list(filter(lambda l: not isinstance(l, DenseLayer), sc.inner_layers))
    assert all(isinstance(l, HadamardLayer) for l in prod_layers)
    assert all(l.num_input_units == prod_num_units for l in dense_layers if sc.layer_outputs(l))
    assert all(l.num_output_units == prod_num_units for l in dense_layers if sc.layer_outputs(l))


@pytest.mark.parametrize(
    "num_units,input_layer",
    itertools.product([1, 3], ["bernoulli", "gaussian"]),
)
def test_multiply_integrate_circuits(num_units: int, input_layer: str):
    sc1 = build_multivariate_monotonic_structured_cpt_pc(
        num_units=num_units, input_layer=input_layer
    )
    sc2 = build_multivariate_monotonic_structured_cpt_pc(
        num_units=num_units * 2 + 1, input_layer=input_layer
    )
    prod_num_units = num_units * (num_units * 2 + 1)
    sc = SF.multiply(sc1, sc2)
    int_sc = SF.integrate(sc)
    assert not int_sc.scope
    assert len(list(int_sc.inputs)) == len(list(sc.inputs))
    assert all(isinstance(isl, LogPartitionLayer) for isl in int_sc.inputs)
    assert len(list(int_sc.inner_layers)) == len(list(sc.inner_layers))
    dense_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
    assert all(isinstance(l.weight.output, KroneckerParameter) for l in dense_layers)
    assert all(
        l.weight.shape == (prod_num_units, prod_num_units)
        for l in dense_layers
        if sc.layer_outputs(l)
    )
    prod_layers = list(filter(lambda l: not isinstance(l, DenseLayer), sc.inner_layers))
    assert all(isinstance(l, HadamardLayer) for l in prod_layers)
    assert all(l.num_input_units == prod_num_units for l in dense_layers if sc.layer_outputs(l))
    assert all(l.num_output_units == prod_num_units for l in dense_layers if sc.layer_outputs(l))


@pytest.mark.parametrize(
    "num_units,input_layer",
    itertools.product([1, 3], ["bernoulli", "gaussian"]),
)
def test_conjugate_circuit(num_units: int, input_layer: str):
    sc1 = build_multivariate_monotonic_structured_cpt_pc(
        num_units=num_units, input_layer=input_layer
    )
    sc = SF.conjugate(sc1)
    assert len(list(sc.inputs)) == len(list(sc.inputs))
    assert len(list(sc.inner_layers)) == len(list(sc.inner_layers))
    dense_layers = list(filter(lambda l: isinstance(l, DenseLayer), sc.inner_layers))
    assert all(isinstance(l.weight.output, ConjugateParameter) for l in dense_layers)
    assert all(
        l.weight.shape == (num_units, num_units) for l in dense_layers if sc.layer_outputs(l)
    )
    prod_layers = list(filter(lambda l: not isinstance(l, DenseLayer), sc.inner_layers))
    assert all(isinstance(l, HadamardLayer) for l in prod_layers)
    assert all(l.num_input_units == num_units for l in dense_layers if sc.layer_outputs(l))
    assert all(l.num_output_units == num_units for l in dense_layers if sc.layer_outputs(l))
