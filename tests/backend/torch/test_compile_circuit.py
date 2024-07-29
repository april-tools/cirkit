import itertools
from typing import cast

import numpy as np
import pytest
import torch
from scipy import integrate

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import TorchDenseLayer, TorchMixingLayer
from cirkit.pipeline import PipelineContext, compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import NormalInitializer
from cirkit.symbolic.layers import GaussianLayer, MixingLayer
from cirkit.symbolic.parameters import Parameter, SoftmaxParameter, TensorParameter
from cirkit.templates.region_graph import RandomBinaryTree
from tests.floats import allclose, isclose
from tests.symbolic.test_utils import build_simple_pc

# TODO: group common code in some utility functions for testing


def copy_folded_parameters(sc: Circuit, compiler: TorchCompiler, fold_compiler: TorchCompiler):
    def copy_parameters(p: Parameter):
        ordering = p.topological_ordering()
        for n in ordering:
            if isinstance(n, TensorParameter):
                p1_t, _ = compiler.state.retrieve_compiled_parameter(n)
                p2_t, fold_idx = fold_compiler.state.retrieve_compiled_parameter(n)
                p1_t._ptensor.copy_(p2_t._ptensor[fold_idx].unsqueeze(dim=0))

    # Set the same parameters of the unfolded circuit
    for sl in sc.layers:
        for _, p in sl.params.items():
            copy_parameters(p)


@pytest.mark.parametrize(
    "fold,num_variables,num_input_units,num_sum_units,num_repetitions",
    itertools.product([False, True], [1, 12], [1, 4], [1, 3], [1, 3]),
)
def test_compile_output_shape(
    fold: bool, num_variables: int, num_input_units: int, num_sum_units: int, num_repetitions: int
):
    compiler = TorchCompiler(fold=fold)
    sc = build_simple_pc(
        num_variables, num_input_units, num_sum_units, num_repetitions=num_repetitions
    )
    tc: TorchCircuit = compiler.compile(sc)

    batch_size = 42
    input_shape = (batch_size, 1, num_variables)
    x = torch.zeros(input_shape, dtype=torch.int64)
    y = tc(x)
    assert y.shape == (batch_size, 1, 1)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize(
    "fold",
    [False, True],
)
def test_modules_parameters(fold: bool):
    compiler = TorchCompiler(fold=fold)
    sc = build_simple_pc(12, 3, 3)
    tc: TorchCircuit = compiler.compile(sc)
    modules = list(tc.modules())
    assert len(modules) > 1
    # print(modules)
    parameters = list(tc.parameters())
    assert len(parameters) > 1
    # print(parameters)
    tc.to("meta")


@pytest.mark.parametrize(
    "fold,semiring,num_variables,normalized",
    itertools.product([False, True], ["lse-sum", "sum-product"], [1, 2, 5], [False, True]),
)
def test_compile_integrate_pc_discrete(
    fold: bool, semiring: str, num_variables: int, normalized: bool
):
    compiler = TorchCompiler(fold=fold, semiring=semiring)
    sc = build_simple_pc(num_variables, 3, 2, num_repetitions=3, normalized=normalized)

    int_sc = SF.integrate(sc)
    int_tc: TorchConstantCircuit = compiler.compile(int_sc)
    assert isinstance(int_tc, TorchConstantCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert isclose(z.item(), 1.0)
        elif semiring == "lse-sum":
            assert isclose(z.item(), 0.0)
        else:
            assert False
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2**num_variables, 1, 1)
    scores = scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(scores), z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(scores, dim=0), z)
    else:
        assert False


@pytest.mark.parametrize(
    "semiring,num_variables,normalized",
    itertools.product(["lse-sum", "sum-product"], [1, 2, 5], [False, True]),
)
def test_compile_integrate_pc_discrete_folded(semiring: str, num_variables: int, normalized: bool):
    compiler = TorchCompiler(fold=False, semiring=semiring)
    sc = build_simple_pc(num_variables, 3, 2, num_repetitions=3, normalized=normalized)
    tc: TorchCircuit = compiler.compile(sc)
    assert isinstance(tc, TorchCircuit)
    int_sc = SF.integrate(sc)
    int_tc: TorchConstantCircuit = compiler.compile(int_sc)
    assert isinstance(int_tc, TorchConstantCircuit)

    fold_compiler = TorchCompiler(fold=True, semiring=semiring)
    folded_tc: TorchCircuit = fold_compiler.compile(sc)
    assert isinstance(folded_tc, TorchCircuit)
    folded_int_sc = SF.integrate(sc)
    folded_int_tc: TorchConstantCircuit = fold_compiler.compile(folded_int_sc)
    assert isinstance(folded_int_tc, TorchConstantCircuit)

    copy_folded_parameters(sc, compiler, fold_compiler)

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert isclose(z.item(), 1.0)
        elif semiring == "lse-sum":
            assert isclose(z.item(), 0.0)
        else:
            assert False
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2**num_variables, 1, 1)
    scores = scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(scores), z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(scores, dim=0), z)
    else:
        assert False

    # Test the partition function value
    folded_z = folded_int_tc()
    assert folded_z.shape == (1, 1)
    folded_z = folded_z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert isclose(folded_z.item(), 1.0)
        elif semiring == "lse-sum":
            assert isclose(folded_z.item(), 0.0)
        else:
            assert False
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    folded_scores = folded_tc(worlds)
    assert folded_scores.shape == (2**num_variables, 1, 1)
    folded_scores = folded_scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(folded_scores), folded_z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(folded_scores, dim=0), folded_z)
    else:
        assert False

    assert allclose(z, folded_z)
    assert allclose(scores, folded_scores)


@pytest.mark.parametrize(
    "fold,semiring,normalized,num_variables,num_products",
    itertools.product(
        [False, True], ["sum-product", "lse-sum"], [False, True], [1, 2, 5], [2, 3, 4]
    ),
)
def test_compile_product_integrate_pc_discrete(
    fold: bool, semiring: str, normalized: bool, num_variables: int, num_products: int
):
    compiler = TorchCompiler(fold=fold, semiring=semiring)
    scs, tcs = [], []
    last_sc = None
    for i in range(num_products):
        sci = build_simple_pc(num_variables, 3 + i, 2 + i, normalized=normalized)
        tci = compiler.compile(sci)
        scs.append(sci)
        tcs.append(tci)
        last_sc = sci if i == 0 else SF.multiply(last_sc, sci)
    tc: TorchCircuit = compiler.compile(last_sc)
    int_sc = SF.integrate(last_sc)
    int_tc = compiler.compile(int_sc)

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert 0.0 < z.item() < 1.0
        elif semiring == "lse-sum":
            assert -np.inf < z.item() < 0.0
        else:
            assert False
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2**num_variables, 1, 1)
    scores = scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(scores, dim=0), z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(scores, dim=0), z)
    else:
        assert False

    # Test the products of the circuits evaluated over all possible assignments
    each_tc_scores = torch.stack([tci(worlds).squeeze() for tci in tcs], dim=0)

    # TODO: should use 'compiler.semiring.prod(each_tc_scores, dim=0)' instead
    if semiring == "sum-product":
        assert allclose(torch.prod(each_tc_scores, dim=0), scores)
    elif semiring == "lse-sum":
        assert allclose(torch.sum(each_tc_scores, dim=0), scores)
    else:
        assert False


@pytest.mark.parametrize(
    "semiring,normalized,num_variables,num_products",
    itertools.product(["sum-product", "lse-sum"], [False, True], [1, 2, 5], [2, 3, 4]),
)
def test_compile_product_integrate_pc_discrete_folded(
    semiring: str, normalized: bool, num_variables: int, num_products: int
):
    compiler = TorchCompiler(fold=False, semiring=semiring)
    scs, tcs = [], []
    last_sc = None
    for i in range(num_products):
        sci = build_simple_pc(num_variables, 3 + i, 2 + i, normalized=normalized)
        tci = compiler.compile(sci)
        scs.append(sci)
        tcs.append(tci)
        last_sc = sci if i == 0 else SF.multiply(last_sc, sci)
    tc: TorchCircuit = compiler.compile(last_sc)
    int_sc = SF.integrate(last_sc)
    int_tc = compiler.compile(int_sc)

    fold_compiler = TorchCompiler(fold=True, semiring=semiring)
    folded_tc: TorchCircuit = fold_compiler.compile(last_sc)
    folded_int_sc = SF.integrate(last_sc)
    folded_int_tc = fold_compiler.compile(folded_int_sc)
    folded_tcs = []
    for sci in scs:
        copy_folded_parameters(sci, compiler, fold_compiler)
        folded_tcs.append(fold_compiler.get_compiled_circuit(sci))

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert 0.0 < z.item() < 1.0
        elif semiring == "lse-sum":
            assert -np.inf < z.item() < 0.0
        else:
            assert False
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2**num_variables, 1, 1)
    scores = scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(scores, dim=0), z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(scores, dim=0), z)
    else:
        assert False

    # Test the products of the circuits evaluated over all possible assignments
    each_tc_scores = torch.stack([tci(worlds).squeeze() for tci in tcs], dim=0)

    # TODO: should use 'compiler.semiring.prod(each_tc_scores, dim=0)' instead
    if semiring == "sum-product":
        assert allclose(torch.prod(each_tc_scores, dim=0), scores)
    elif semiring == "lse-sum":
        assert allclose(torch.sum(each_tc_scores, dim=0), scores)
    else:
        assert False

    # Test the partition function value
    folded_z = folded_int_tc()
    assert folded_z.shape == (1, 1)
    folded_z = folded_z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert 0.0 < folded_z.item() < 1.0
        elif semiring == "lse-sum":
            assert -np.inf < folded_z.item() < 0.0
        else:
            assert False
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    folded_scores = folded_tc(worlds)
    assert folded_scores.shape == (2**num_variables, 1, 1)
    folded_scores = folded_scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(folded_scores, dim=0), z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(folded_scores, dim=0), z)
    else:
        assert False

    # Test the products of the circuits evaluated over all possible assignments
    folded_each_tc_scores = torch.stack(
        [folded_tci(worlds).squeeze() for folded_tci in folded_tcs], dim=0
    )

    # TODO: should use 'compiler.semiring.prod(each_tc_scores, dim=0)' instead
    if semiring == "sum-product":
        assert allclose(torch.prod(folded_each_tc_scores, dim=0), folded_scores)
    elif semiring == "lse-sum":
        assert allclose(torch.sum(folded_each_tc_scores, dim=0), folded_scores)
    else:
        assert False

    assert allclose(each_tc_scores, folded_each_tc_scores)
    assert allclose(z, folded_z)


@pytest.mark.slow
@pytest.mark.parametrize(
    "fold",
    [False, True],
)
def test_compile_integrate_pc_continuous(fold: bool):
    compiler = TorchCompiler(semiring="lse-sum", fold=fold)
    num_variables = 2
    sc = build_simple_pc(num_variables, input_layer="gaussian")

    int_sc = SF.integrate(sc)
    int_tc: TorchConstantCircuit = compiler.compile(int_sc)
    assert isinstance(int_tc, TorchConstantCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()

    # Test the integral of the circuit (using a quadrature rule)
    df = lambda y, x: torch.exp(tc(torch.Tensor([[[x, y]]]))).squeeze()
    int_a, int_b = -np.inf, np.inf
    ig, err = integrate.dblquad(df, int_a, int_b, int_a, int_b)
    assert np.isclose(ig, torch.exp(z).item(), atol=1e-15)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_products",
    [2, 3],
)
def test_compile_product_integrate_pc_continuous(num_products: int):
    compiler = TorchCompiler(semiring="lse-sum", fold=True)
    scs, tcs = [], []
    last_sc = None
    for i in range(num_products):
        sci = build_simple_pc(2, 2 + i, 2, input_layer="gaussian")
        tci = compiler.compile(sci)
        scs.append(sci)
        tcs.append(tci)
        last_sc = sci if i == 0 else SF.multiply(last_sc, sci)
    tc: TorchCircuit = compiler.compile(last_sc)
    int_sc = SF.integrate(last_sc)
    int_tc = compiler.compile(int_sc)

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()

    # Test the integral of the circuit (using a quadrature rule)
    df = lambda y, x: torch.exp(tc(torch.Tensor([[[x, y]]]))).squeeze()
    int_a, int_b = -np.inf, np.inf
    ig, err = integrate.dblquad(df, int_a, int_b, int_a, int_b)
    assert np.isclose(ig, torch.exp(z).item(), atol=1e-15)

    # Test the products of the circuits evaluated over all possible assignments
    xs = torch.linspace(-5, 5, steps=16)
    ys = torch.linspace(-5, 5, steps=16)
    points = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=1).view(-1, 1, 2)
    scores = tc(points)
    scores = scores.squeeze()
    each_tc_scores = torch.stack([tci(points).squeeze() for tci in tcs], dim=0)
    assert allclose(torch.sum(each_tc_scores, dim=0), scores)


def test_compile_circuit_cp_layers():
    rg = RandomBinaryTree(8, depth=3, num_repetitions=4)
    symbolic_circuit = Circuit.from_region_graph(
        rg,
        input_factory=lambda scope, num_units, num_channels: GaussianLayer(
            scope, num_units, num_channels=num_channels
        ),
        sum_product="cp",
        dense_weight_factory=lambda shape: Parameter.from_unary(
            SoftmaxParameter(shape), TensorParameter(*shape, initializer=NormalInitializer())
        ),
        mixing_factory=lambda scope, num_units, arity: MixingLayer(scope, num_units, arity=arity),
        num_input_units=64,
        num_sum_units=64,
    )

    pipeline = PipelineContext(backend="torch", fold=True, optimize=True)
    with pipeline:
        circuit = cast(TorchCircuit, compile(symbolic_circuit))
    y = circuit(torch.randn(42, 1, 8))
    assert y.shape == (42, 1, 1)

    # for l in circuit.layers:
    #     if isinstance(l, (TorchDenseLayer, TorchMixingLayer)):
    #         print(l.__class__.__name__, (l.weight.num_folds, *l.weight.shape))

    layers = list(
        filter(lambda l: isinstance(l, (TorchDenseLayer, TorchMixingLayer)), circuit.layers)
    )
    assert (layers[0].weight.num_folds, *layers[0].weight.shape) == (32, 64, 64)
    assert (layers[1].weight.num_folds, *layers[1].weight.shape) == (16, 64, 64)
    assert (layers[2].weight.num_folds, *layers[2].weight.shape) == (8, 1, 64)
    assert (layers[3].weight.num_folds, *layers[3].weight.shape) == (1, 1, 4)
