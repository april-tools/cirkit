import itertools

import pytest
import torch

pytest.importorskip("extended_einsum")

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.xe.circuits import XETorchCircuit
from cirkit.backend.xe.compiler import XETorchCompiler
from cirkit.backend.xe.layers import XESumLayer
from tests.backend.xe.test_utils import copy_parameters
from tests.symbolic.test_utils import build_multivariate_monotonic_structured_cpt_pc


def compile_and_align_circuits(
    sc, *, semiring: str, fold: bool, optimize: bool, **backend_kwargs
) -> tuple:
    """Compile a symbolic circuit with both the extended-einsum and the torch
    compilers, and copy the parameter values of the former into the latter."""
    xe_compiler = XETorchCompiler(semiring=semiring, fold=fold, optimize=optimize, **backend_kwargs)
    torch_compiler = TorchCompiler(semiring=semiring, fold=True, optimize=True)
    xc = xe_compiler.compile(sc)
    tc = torch_compiler.compile(sc)
    copy_parameters(sc, xe_compiler, torch_compiler)
    return xc, tc


@pytest.mark.parametrize(
    "semiring,fold,optimize,product_layer",
    itertools.product(
        ["lse-sum", "sum-product"], [False, True], [False, True], ["hadamard", "kronecker"]
    ),
)
def test_compile_circuit_parity(semiring: str, fold: bool, optimize: bool, product_layer: str):
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2, product_layer=product_layer)
    xc, tc = compile_and_align_circuits(sc, semiring=semiring, fold=fold, optimize=optimize)
    assert isinstance(xc, XETorchCircuit)
    assert xc.num_variables == tc.num_variables == 5
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=5)))
    xy = xc(worlds)
    ty = tc(worlds)
    assert xy.shape == ty.shape == (32, 1, 1)
    assert torch.all(torch.isfinite(xy))
    assert torch.allclose(xy, ty, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "stability",
    ["unstable", "logspace_min", "logspace_max", "scaled_min", "scaled_max", "scaled_sum"],
)
def test_compile_circuit_stability_modes(stability: str):
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2)
    xc, tc = compile_and_align_circuits(
        sc, semiring="lse-sum", fold=True, optimize=True, stability=stability
    )
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=5)))
    assert torch.allclose(xc(worlds), tc(worlds), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fold_depth", ["input", "output"])
def test_compile_circuit_fold_depths(fold_depth: str):
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2)
    xc, tc = compile_and_align_circuits(
        sc, semiring="lse-sum", fold=True, optimize=True, fold_depth=fold_depth
    )
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=5)))
    assert torch.allclose(xc(worlds), tc(worlds), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("input_layer", ["gaussian", "embedding"])
def test_compile_circuit_input_layers(input_layer: str):
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2, input_layer=input_layer)
    # Note that we use the sum-product semiring for the embedding input layers,
    # since they encode possibly-negative functions
    semiring = "lse-sum" if input_layer == "gaussian" else "sum-product"
    xc, tc = compile_and_align_circuits(sc, semiring=semiring, fold=True, optimize=True)
    if input_layer == "gaussian":
        x = torch.randn(11, 5)
    else:
        x = torch.randint(2, size=(11, 5))
    assert torch.allclose(xc(x), tc(x), rtol=1e-5, atol=1e-6)


def test_folded_frontend_uses_xe_input_order():
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2)
    xc = XETorchCompiler(semiring="lse-sum", fold=True).compile(sc)
    executor = xc._executor(11, torch.device("cpu"))  # pylint: disable=protected-access

    assert all(
        getattr(runtime_input, "axis0_order", None) is None
        for runtime_input in executor._runtime_inputs  # pylint: disable=protected-access
    )


def test_folded_softmax_weights_are_lowered_into_xe():
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2, normalized=True)
    xc = XETorchCompiler(semiring="lse-sum", fold=True).compile(sc)
    sum_layers = [layer for layer in xc.layers if isinstance(layer, XESumLayer)]

    assert sum_layers
    assert all(layer.weight_input is not None for layer in sum_layers)
    assert all(layer.weight_softmax_dim is not None for layer in sum_layers)


def test_unsupported_weight_parameter_uses_torch_fallback():
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2, normalized=False)
    xc, tc = compile_and_align_circuits(sc, semiring="sum-product", fold=True, optimize=True)
    sum_layers = [layer for layer in xc.layers if isinstance(layer, XESumLayer)]
    x = torch.randint(2, size=(11, 5))

    assert sum_layers
    assert all(layer.weight_input is None for layer in sum_layers)
    assert all(layer.weight_softmax_dim is None for layer in sum_layers)
    assert torch.allclose(xc(x), tc(x), rtol=1e-5, atol=1e-6)


def test_compile_circuit_batch_sizes():
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2)
    compiler = XETorchCompiler(semiring="lse-sum", fold=True, optimize=True)
    xc = compiler.compile(sc)
    x = torch.randint(2, size=(13, 5))
    # The circuit lowers one extended-einsum program for each batch size, and the
    # evaluations must be consistent across batch sizes
    y = xc(x)
    assert y.shape == (13, 1, 1)
    assert torch.allclose(xc(x[:1]), y[:1], rtol=1e-6, atol=1e-7)
    assert torch.allclose(xc(x[:7]), y[:7], rtol=1e-6, atol=1e-7)


def test_compile_circuit_gradients_and_training():
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2)
    compiler = XETorchCompiler(semiring="lse-sum", fold=True, optimize=True)
    xc = compiler.compile(sc)
    parameters = [p for p in xc.parameters() if p.requires_grad]
    assert parameters
    data = torch.randint(2, size=(32, 5))
    with torch.enable_grad():
        loss = -torch.mean(xc(data))
        loss.backward()
        for p in parameters:
            assert p.grad is not None
            assert torch.all(torch.isfinite(p.grad))
        optimizer = torch.optim.Adam(xc.parameters(), lr=0.05)
        initial_loss = loss.item()
        for _ in range(16):
            optimizer.zero_grad()
            loss = -torch.mean(xc(data))
            loss.backward()
            optimizer.step()
    assert loss.item() < initial_loss


def test_compiler_symbolic_compiled_circuit_maps():
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2)
    compiler = XETorchCompiler()
    assert not compiler.is_compiled(sc)
    xc = compiler.compile(sc)
    assert compiler.is_compiled(sc)
    assert compiler.has_symbolic(xc)
    assert compiler.get_compiled_circuit(sc) is xc
    assert compiler.get_symbolic_circuit(xc) is sc
    # Compiling the same symbolic circuit again returns the same compiled circuit
    assert compiler.compile(sc) is xc


def test_compiler_invalid_flags():
    with pytest.raises(ValueError, match="semiring"):
        XETorchCompiler(semiring="max-product")
    with pytest.raises(ValueError, match="folding depth"):
        XETorchCompiler(fold_depth="middle")
    with pytest.raises(ValueError, match="scale interval"):
        XETorchCompiler(scale_interval=0)
