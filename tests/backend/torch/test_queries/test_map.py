import itertools

import numpy as np
import pytest
import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.queries import MAPQuery
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import NormalInitializer
from cirkit.symbolic.layers import CategoricalLayer, HadamardLayer, SumLayer
from cirkit.symbolic.parameters import (
    ConstantParameter,
    Parameter,
    SoftmaxParameter,
    TensorParameter,
)
from cirkit.templates import data_modalities, pgms, utils
from cirkit.utils.scope import Scope
from tests.floats import allclose


def build_deterministic_categorical_mixture(num_units: int):
    true_p = np.array([[0.0, 1.0]] * num_units)
    false_p = np.array([[1.0, 0.0]] * num_units)

    # construct simple deterministic circuit
    Xt = CategoricalLayer(
        Scope([0]),
        num_units,
        num_categories=2,
        probs=Parameter.from_input(ConstantParameter(num_units, 2, value=true_p)),
    )
    Xf = CategoricalLayer(
        Scope([0]),
        num_units,
        num_categories=2,
        probs=Parameter.from_input(ConstantParameter(num_units, 2, value=false_p)),
    )

    Yt = CategoricalLayer(
        Scope([1]),
        num_units,
        num_categories=2,
        probs=Parameter.from_input(ConstantParameter(num_units, 2, value=true_p)),
    )
    Yf = CategoricalLayer(
        Scope([1]),
        num_units,
        num_categories=2,
        probs=Parameter.from_input(ConstantParameter(num_units, 2, value=false_p)),
    )

    prod_Xf_Yf = HadamardLayer(num_units, 2)
    prod_Xt_Yf = HadamardLayer(num_units, 2)
    prod_Xt_Yt = HadamardLayer(num_units, 2)

    dense_weight_factory = lambda shape: Parameter.from_unary(
        SoftmaxParameter(shape, axis=1),
        TensorParameter(*shape, initializer=NormalInitializer()),
    )
    sum = SumLayer(num_units, num_units, 2, weight_factory=dense_weight_factory)
    root = SumLayer(num_units, 1, 2, weight_factory=dense_weight_factory)

    in_nodes = {
        root: [prod_Xf_Yf, sum],
        prod_Xf_Yf: [Xf, Yf],
        sum: [prod_Xt_Yf, prod_Xt_Yt],
        prod_Xt_Yf: [Xt, Yf],
        prod_Xt_Yt: [Xt, Yt],
    }

    sc = Circuit(
        layers=[root, sum, prod_Xf_Yf, prod_Xt_Yf, prod_Xt_Yt, Xt, Xf, Yt, Yf],
        in_layers=in_nodes,
        outputs=[
            root,
        ],
    )

    return sc


@pytest.mark.parametrize(
    "semiring,fold,optimize,num_units",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True], [1, 20]),
)
def test_query_map(semiring: str, fold: bool, optimize: bool, num_units: int):
    sc = build_deterministic_categorical_mixture(num_units)
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    # The following function computes a circuit where we have computed the
    # partition function and a marginal by hand.

    tc: TorchCircuit = compiler.compile(sc)

    # compute the likelihood for all possible assignments
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=sc.num_variables)), dtype=torch.long
    )
    worlds_ll = tc(worlds)

    # compute map and check that its state matches the enumerated one
    map_query = MAPQuery(tc)
    map_value, map_state = map_query()

    if num_units == 1:
        # deterministic circuit: check correctness of query
        assert allclose(worlds[worlds_ll.argmax()], map_state)
    else:
        # TODO: Figure out a way of checking approximate correctness
        pass


@pytest.mark.parametrize(
    "semiring,fold,optimize,num_units",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True], [1, 20]),
)
def test_query_map_marginalized(semiring: str, fold: bool, optimize: bool, num_units: int):
    sc = pgms.deterministic_fully_factorized(3)
    sc = SF.integrate(sc, Scope([0]))
    
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    # The following function computes a circuit where we have computed the
    # partition function and a marginal by hand.

    tc: TorchCircuit = compiler.compile(sc)

    # compute the likelihood for all possible assignments
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=3)), dtype=torch.long
    )
    worlds_ll = tc(worlds)

    # compute map and check that its state matches the enumerated one
    map_query = MAPQuery(tc)
    map_value, map_state = map_query()

@pytest.mark.parametrize(
    "semiring,fold,optimize,num_units",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True], [1, 20]),
)
def test_query_map_with_evidence(semiring: str, fold: bool, optimize: bool, num_units: int):
    sc = build_deterministic_categorical_mixture(num_units)
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    # The following function computes a circuit where we have computed the
    # partition function and a marginal by hand.

    tc: TorchCircuit = compiler.compile(sc)

    # compute the likelihood for all possible assignments
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=sc.num_variables)), dtype=torch.long
    )
    worlds_ll = tc(worlds)

    # compute map and check that its state matches the enumerated one
    map_query = MAPQuery(tc)
    map_value, map_state = map_query(
        x=torch.tensor([[0, 1]]), evidence_vars=torch.tensor([[False, True]])
    )

    if num_units == 1:
        # deterministic circuit: check correctness of query
        assert allclose(map_state, torch.tensor([1, 1]))
    else:
        # TODO: Figure out a way of checking approximate correctness
        pass


@pytest.mark.parametrize(
    "semiring,fold,optimize,num_units",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True], [1, 20]),
)
def test_query_map_image_data(semiring, fold, optimize, num_units):
    symbolic_circuit = data_modalities.image_data(
        (1, 10, 10),  # The shape of MNIST image, i.e., (num_channels, image_height, image_width)
        region_graph="quad-graph",  # Select the structure of the circuit to follow the QuadGraph region graph
        input_layer="categorical",  # Use Categorical distributions for the pixel values (0-255) as input layers
        num_input_units=num_units,  # Each input layer consists of 64 Categorical input units
        sum_product_layer="cp",  # Use CP sum-product layers, i.e., alternate dense layers with Hadamard product layers
        num_sum_units=num_units,  # Each dense sum layer consists of 64 sum units
        sum_weight_param=utils.Parameterization(
            activation="softmax",  # Parameterize the sum weights by using a softmax activation
            initialization="normal",  # Initialize the sum weights by sampling from a standard normal distribution
        ),
    )

    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)

    tc = compiler.compile(symbolic_circuit).train()
    map_query = MAPQuery(tc)
    map_query()


@pytest.mark.parametrize(
    "semiring,fold,optimize,num_units",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True], [1, 20]),
)
def test_query_map_conditional_circuit(semiring, fold, optimize, num_units):
    sc = build_deterministic_categorical_mixture(1)

    pm = {"sum": list(sc.sum_layers)}
    c_sc, gf_specs = SF.condition_circuit(sc, gate_functions=pm)

    ctx = PipelineContext(backend="torch", semiring=semiring, fold=fold, optimize=optimize)
    ctx.add_gate_function("sum.weight.0", lambda x: x.view(-1, *gf_specs["sum.weight.0"]))
    tc = ctx.compile(c_sc)

    # compute map and check that its state matches the enumerated one
    map_query = MAPQuery(tc)
    map_query(
        gate_function_kwargs={"sum.weight.0": {"x": torch.rand(2, *gf_specs["sum.weight.0"])}}
    )


@pytest.mark.parametrize(
    "semiring,fold,optimize,num_units",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True], [1, 20]),
)
def test_query_map_conditional_circuit_with_evidence(semiring, fold, optimize, num_units):
    sc = build_deterministic_categorical_mixture(1)

    pm = {"sum": list(sc.sum_layers)}
    c_sc, gf_specs = SF.condition_circuit(sc, gate_functions=pm)

    ctx = PipelineContext(backend="torch", semiring=semiring, fold=fold, optimize=optimize)
    ctx.add_gate_function("sum.weight.0", lambda x: x.view(-1, *gf_specs["sum.weight.0"]))
    tc = ctx.compile(c_sc)

    # compute map and check that its state matches the enumerated one
    map_query = MAPQuery(tc)
    map_query(
        x=torch.tensor([[0, 1], [0, 1]]),
        evidence_vars=torch.tensor([[False, True], [False, True]]),
        gate_function_kwargs={"sum.weight.0": {"x": torch.rand(2, *gf_specs["sum.weight.0"])}},
    )

@pytest.mark.parametrize(
    "semiring,fold,optimize,num_units",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True], [1, 20]),
)
def test_query_map_marginalized_conditional_circuit(semiring, fold, optimize, num_units):
    sc = pgms.deterministic_fully_factorized(3)

    pm = {"sum": list(sc.sum_layers)}
    c_sc, gf_specs = SF.condition_circuit(sc, gate_functions=pm)
    c_sc = SF.integrate(sc, Scope([0]))

    ctx = PipelineContext(backend="torch", semiring=semiring, fold=fold, optimize=optimize)
    ctx.add_gate_function("sum.weight.0", lambda x: x.view(-1, *gf_specs["sum.weight.0"]))
    tc = ctx.compile(c_sc)

    # compute map and check that its state matches the enumerated one
    map_query = MAPQuery(tc)
    map_query(
        gate_function_kwargs={"sum.weight.0": {"x": torch.rand(2, *gf_specs["sum.weight.0"])}},
    )
