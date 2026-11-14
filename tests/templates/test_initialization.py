"""Numerical and structural regressions for data-driven circuit initialization."""

import functools
import itertools

import numpy as np
import pytest
import torch

from cirkit.backend.torch.layers import TorchInputLayer, TorchSumLayer
from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import CategoricalLayer, MultichannelCategoricalLayer, SumLayer
from cirkit.symbolic.parameters import MixingWeightParameter, mixing_weight_factory
from cirkit.templates.data_modalities import image_data, tabular_data
from cirkit.templates.initialization import DataDrivenInitialization, initialize_parameters
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from cirkit.utils.scope import Scope


@pytest.mark.parametrize(
    "sharing,inner", tuple(itertools.product(["none", "full"], ["cp", "cp-t", "tucker"]))
)
def test_tabular_estimates_match_counts(sharing, inner):
    data = torch.tensor([[0, 0], [0, 1], [0, 1], [1, 1]])
    circuit = tabular_data(
        data=data,
        input_layers={"name": "categorical", "args": {"num_categories": 2}},
        num_input_units=2,
        sum_product_layer=inner,
        num_sum_units=2,
        input_sharing=sharing,
        initialization=DataDrivenInitialization(alpha=2.0, noise_scale=0.0),
    )
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=2)))
    marginals = torch.tensor([[5 / 8, 3 / 8], [3 / 8, 5 / 8]])
    if sharing == "full":
        marginals[:] = 0.5
    expected = marginals[0, worlds[:, 0]] * marginals[1, worlds[:, 1]]
    ctx = PipelineContext(backend="torch", semiring="sum-product", fold=True, optimize=True)
    compiled = ctx.compile(circuit)
    assert circuit.is_smooth and circuit.is_decomposable
    torch.testing.assert_close(compiled(worlds).flatten(), expected)
    torch.testing.assert_close(compiled(worlds).sum(), torch.tensor(1.0))


@pytest.mark.parametrize("mixing", [False, True])
def test_branch_estimates_preserve_mixing_structure(mixing):
    leaves = [CategoricalLayer(Scope([0]), 2, num_categories=2) for _ in range(2)]
    factory = parameterization_to_factory(Parameterization(activation="softmax"))
    if mixing:
        factory = functools.partial(mixing_weight_factory, param_factory=factory)
    root = SumLayer(2, 2, arity=2, weight_factory=factory)
    circuit = Circuit([*leaves, root], {root: leaves}, [root])
    initialize_parameters(
        circuit,
        torch.tensor([[0], [0], [0], [1]]),
        initialization=DataDrivenInitialization(alpha=0.0, noise_scale=0.0, seed=0),
        sum_weight_param=Parameterization(activation="softmax"),
    )
    assert isinstance(root.weight.output, MixingWeightParameter) == mixing
    compiled = PipelineContext(
        backend="torch", semiring="lse-sum", fold=False, optimize=False
    ).compile(circuit)
    weights = next(layer.weight() for layer in compiled.layers if isinstance(layer, TorchSumLayer))[
        0
    ]
    if mixing:
        assert torch.count_nonzero(weights) == 4
        torch.testing.assert_close(weights[0, [1, 3]], torch.zeros(2))
        torch.testing.assert_close(weights[1, [0, 2]], torch.zeros(2))
    else:
        assert torch.count_nonzero(weights) == 8
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    scores = compiled(torch.tensor([[0], [1]])).exp()
    torch.testing.assert_close(scores[:, 0, 0], torch.tensor([0.75, 0.25]))


def test_shared_dag_leaf_receives_every_parent_population():
    leaf = CategoricalLayer(Scope([0]), 1, num_categories=2)
    branches = [SumLayer(1, 1) for _ in range(2)]
    root = SumLayer(1, 1, arity=2)
    circuit = Circuit(
        [leaf, *branches, root], {root: branches, **{b: [leaf] for b in branches}}, [root]
    )
    initialize_parameters(
        circuit,
        torch.tensor([[0], [0], [0], [1]]),
        initialization=DataDrivenInitialization(alpha=0.0, noise_scale=0.0),
        sum_weight_param=Parameterization(activation="softmax"),
    )
    compiled = PipelineContext(backend="torch", semiring="sum-product").compile(circuit)
    torch.testing.assert_close(
        compiled(torch.tensor([[0], [1]])).flatten(), torch.tensor([0.75, 0.25])
    )


def test_shared_dag_populations_do_not_duplicate_observations():
    leaf = CategoricalLayer(Scope([0]), 1, num_categories=2)
    roots = [SumLayer(1, 1) for _ in range(2)]
    circuit = Circuit([leaf, *roots], {root: [leaf] for root in roots}, roots)
    initialize_parameters(
        circuit,
        torch.tensor([[0], [0], [0], [1]]),
        initialization=DataDrivenInitialization(alpha=2.0, noise_scale=0.0),
        sum_weight_param=Parameterization(activation="softmax"),
    )
    compiled = PipelineContext(backend="torch", semiring="sum-product").compile(circuit)
    expected = torch.tensor([5 / 8, 3 / 8])[:, None, None].expand(2, 2, 1)
    torch.testing.assert_close(compiled(torch.tensor([[0], [1]])), expected)


@pytest.mark.parametrize("fold,optimize", tuple(itertools.product([False, True], repeat=2)))
def test_rgb_sharing_pools_pixels_per_channel(fold, optimize):
    data = torch.tensor(
        [[[[0, 0]], [[0, 1]], [[1, 1]]], [[[0, 1]], [[1, 1]], [[1, 0]]]], dtype=torch.uint8
    )
    symbolic = image_data(
        (3, 1, 2),
        region_graph="quad-tree-2",
        input_layer="categorical",
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        input_sharing="full",
        data=data,
        initialization=DataDrivenInitialization(alpha=0.0, noise_scale=0.0),
    )
    assert all(isinstance(layer, MultichannelCategoricalLayer) for layer in symbolic.input_layers)
    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=fold, optimize=optimize)
    compiled = ctx.compile(symbolic)
    expected = torch.tensor([0.75**2 * 0.25 * 0.75 * 0.75**2, 0.75 * 0.25 * 0.75**2 * 0.75 * 0.25])
    torch.testing.assert_close(compiled(data.flatten(1)).flatten(), expected.log())
    partition = ctx.compile(functional.integrate(symbolic))
    torch.testing.assert_close(partition().flatten(), torch.zeros(1))
    marginal = ctx.compile(functional.integrate(symbolic, scope=Scope([0, 1])))
    expected_marginal = torch.tensor([0.25 * 0.75**3, 0.25 * 0.75**3])
    torch.testing.assert_close(marginal(data.flatten(1)).flatten(), expected_marginal.log())
    input_params = {
        id(p): p
        for layer in compiled.layers
        if isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    }
    assert sum(p.numel() for p in input_params.values()) == 2 * 3 * 256
    with torch.enable_grad():
        compiled(data.flatten(1)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in input_params.values())


@pytest.mark.parametrize("shape", [(1, 2, 2), (2, 2), (4,)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.float32])
def test_image_layouts_and_seed(shape, dtype):
    data = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0]], dtype=dtype).reshape(3, *shape)
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    kwargs = {
        "image_shape": (1, 2, 2),
        "input_layer": "categorical",
        "num_input_units": 2,
        "sum_product_layer": "cp",
        "num_sum_units": 2,
        "data": data,
        "initialization": DataDrivenInitialization(seed=12),
    }
    first = image_data(**kwargs)
    second = image_data(**kwargs)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    after_state = np.random.get_state()
    assert numpy_state[0] == after_state[0]
    np.testing.assert_array_equal(numpy_state[1], after_state[1])
    assert numpy_state[2:] == after_state[2:]
    first_compiled = PipelineContext(backend="torch").compile(first)
    second_compiled = PipelineContext(backend="torch").compile(second)
    torch.testing.assert_close(first_compiled(data.flatten(1)), second_compiled(data.flatten(1)))


@pytest.mark.parametrize("samples", [torch.zeros(1, 4), torch.zeros(4, 4)])
@pytest.mark.parametrize("activation", ["none", "softmax", "softplus", "sigmoid", "positive-clamp"])
def test_small_and_constant_populations(samples, activation):
    circuit = image_data(
        (1, 2, 2),
        input_layer="categorical",
        num_input_units=2,
        sum_product_layer="tucker",
        num_sum_units=2,
        num_classes=3,
        data=samples,
        initialization=DataDrivenInitialization(alpha=0.0, noise_scale=0.0),
        sum_weight_param=Parameterization(activation=activation),
    )
    compiled = PipelineContext(backend="torch", fold=True, optimize=True).compile(circuit)
    result = compiled(samples)
    assert result.shape == (len(samples), 1, 3)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    "data",
    [
        torch.empty(0, 2),
        torch.ones(3, 1),
        torch.tensor([[0.5, 0]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([[2, 0]]),
        torch.tensor([[float("nan"), 0]]),
        torch.tensor([[float("inf"), 0]]),
        torch.ones(1, 2, dtype=torch.complex64),
    ],
)
def test_invalid_categorical_data(data):
    with pytest.raises(ValueError):
        tabular_data(
            num_features=2,
            data=data,
            input_layers={"name": "categorical", "args": {"num_categories": 2}},
            num_input_units=1,
            sum_product_layer="cp",
            num_sum_units=1,
            initialization=DataDrivenInitialization(),
        )


def test_unused_data_preserves_the_legacy_tabular_path():
    """Data is ignored for a fixed structure unless initialization consumes it."""
    circuit = tabular_data(
        num_features=2,
        data=torch.empty(0, 3),
        input_layers={"name": "categorical", "args": {"num_categories": 2}},
        num_input_units=1,
        sum_product_layer="cp",
        num_sum_units=1,
    )
    assert set(circuit.scope) == {0, 1}


@pytest.mark.parametrize(
    "settings",
    [{"alpha": -1}, {"alpha": float("nan")}, {"noise_scale": float("inf")}, {"seed": -1}],
)
def test_invalid_initialization_settings(settings):
    with pytest.raises(ValueError):
        DataDrivenInitialization(**settings)


@pytest.mark.parametrize(
    "options",
    [
        {"data": None},
        {"input_layer": "gaussian"},
        {"input_params": {}},
        {"input_sharing": "unknown"},
        {"region_graph": "unknown"},
        {"sum_product_layer": "unknown"},
        {"sum_weight_param": Parameterization(activation="clamp")},
        {"sum_weight_param": Parameterization(activation="softmax", activation_kwargs={"axis": 0})},
    ],
)
def test_invalid_image_initialization_options(options):
    kwargs = {
        "image_shape": (1, 2, 2),
        "input_layer": "categorical",
        "num_input_units": 1,
        "sum_product_layer": "cp",
        "num_sum_units": 1,
        "data": torch.zeros(2, 4),
        "initialization": DataDrivenInitialization(),
    }
    kwargs.update(options)
    with pytest.raises(ValueError):
        image_data(**kwargs)


def test_tabular_heterogeneous_cardinalities():
    circuit = tabular_data(
        data=torch.tensor([[0, 2], [1, 1]]),
        input_layers=[{"name": "categorical", "args": {"num_categories": n}} for n in (2, 3)],
        num_input_units=1,
        sum_product_layer="cp",
        num_sum_units=1,
        initialization=DataDrivenInitialization(noise_scale=0.0),
    )
    worlds = torch.tensor(list(itertools.product(range(2), range(3))))
    compiled = PipelineContext(backend="torch", semiring="sum-product").compile(circuit)
    torch.testing.assert_close(compiled(worlds).sum(), torch.tensor(1.0))


def test_full_sharing_rejects_heterogeneous_cardinalities():
    with pytest.raises(ValueError, match="identical categorical"):
        tabular_data(
            num_features=2,
            input_layers=[{"name": "categorical", "args": {"num_categories": n}} for n in (2, 3)],
            num_input_units=1,
            sum_product_layer="cp",
            num_sum_units=1,
            input_sharing="full",
        )


@pytest.mark.parametrize(
    "sharing,initialized", tuple(itertools.product(["none", "full"], [False, True]))
)
def test_sharing_is_independent_of_initialization(sharing, initialized):
    circuit = tabular_data(
        data=torch.tensor([[0, 1, 1], [1, 0, 0]]),
        input_layers={"name": "categorical", "args": {"num_categories": 2}},
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        input_sharing=sharing,
        initialization=DataDrivenInitialization() if initialized else None,
    )
    compiled = PipelineContext(backend="torch", fold=True, optimize=True).compile(circuit)
    parameters = {
        id(p): p
        for layer in compiled.layers
        if isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    }
    assert sum(p.numel() for p in parameters.values()) == (4 if sharing == "full" else 12)
    assert torch.isfinite(compiled(torch.tensor([[0, 1, 0]]))).all()


@pytest.mark.parametrize("region_graph", ["chow-liu-tree", "random-binary-tree"])
def test_structure_learning_and_parameter_initialization_are_separate(region_graph):
    data = torch.tensor(list(itertools.product([0, 1], repeat=3)))
    circuit = tabular_data(
        region_graph=region_graph,
        data=data,
        input_layers={"name": "categorical", "args": {"num_categories": 2}},
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        initialization=DataDrivenInitialization(noise_scale=0.0),
    )
    compiled = PipelineContext(backend="torch", semiring="sum-product").compile(circuit)
    torch.testing.assert_close(compiled(data).flatten(), torch.full((8,), 1 / 8))


def test_additive_smoothing():
    circuit = tabular_data(
        data=torch.tensor([[0], [0], [0], [1]]),
        input_layers={"name": "categorical", "args": {"num_categories": 2}},
        num_input_units=1,
        sum_product_layer="cp",
        num_sum_units=1,
        initialization=DataDrivenInitialization(alpha=2.0, noise_scale=0.0),
    )
    compiled = PipelineContext(backend="torch", semiring="sum-product").compile(circuit)
    torch.testing.assert_close(compiled(torch.tensor([[0]])).flatten(), torch.tensor([5 / 8]))
