import itertools

import pytest


from cirkit.symbolic.layers import CategoricalLayer, HadamardLayer, SumLayer
from cirkit.templates import pgms
from cirkit.utils.scope import Scope


@pytest.mark.parametrize("num_variables", [1, 5])
def test_pgm_fully_factorized(num_variables: int):
    circuit = pgms.fully_factorized(
        num_variables,
        input_layer='categorical',
        input_layer_kwargs={'num_categories': 3}
    )
    assert circuit.scope == Scope(range(num_variables))
    input_layers = list(circuit.inputs)
    product_layers = list(circuit.product_layers)
    sum_layers = list(circuit.sum_layers)
    assert len(input_layers) == num_variables
    assert all(isinstance(sl, CategoricalLayer) for sl in input_layers)
    assert all(sl.num_output_units == 1 for sl in product_layers)
    assert len(sum_layers) == 0
    assert len(product_layers) == (1 if num_variables > 1 else 0)
    if num_variables > 1:
        prod_sl, = product_layers
        assert isinstance(prod_sl, HadamardLayer)
        assert prod_sl.num_input_units == 1 and prod_sl.arity == num_variables


@pytest.mark.parametrize("num_variables,num_latent_states", itertools.product([1, 5], [1, 6]))
def test_pgm_hmm(num_variables: int, num_latent_states: int):
    ordering = list(range(num_variables))
    circuit = pgms.hmm(
        ordering,
        input_layer='categorical',
        num_latent_states=num_latent_states,
        input_layer_kwargs={'num_categories': 3}
    )
    assert circuit.scope == Scope(range(num_variables))
    input_layers = list(circuit.inputs)
    product_layers = list(circuit.product_layers)
    sum_layers = list(circuit.sum_layers)
    assert len(input_layers) == num_variables
    assert all(isinstance(sl, CategoricalLayer) for sl in input_layers)
    assert len(product_layers) == num_variables - 1
    assert all(isinstance(sl, HadamardLayer) for sl in product_layers)
    assert all(sl.num_input_units == num_latent_states and sl.arity == 2 for sl in product_layers)
    assert len(sum_layers) == num_variables
    assert all(isinstance(sl, SumLayer) for sl in sum_layers)
    assert all(sl.num_input_units == num_latent_states and sl.arity == 1 for sl in sum_layers)
