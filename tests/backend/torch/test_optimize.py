import itertools

import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import (
    TorchCategoricalLayer,
    TorchDenseLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchTuckerLayer,
)
from cirkit.backend.torch.layers.optimized import TorchCPTLayer, TorchTensorDotLayer
from tests.floats import allclose
from tests.symbolic.test_utils import build_simple_pc


def test_optimize_tucker():
    num_variables = 6
    sc = build_simple_pc(num_variables, 3, 2, sum_product_layer="tucker")

    unoptimized_compiler = TorchCompiler(fold=True, optimize=False)
    unoptimized_tc: TorchCircuit = unoptimized_compiler.compile(sc)

    optimized_compiler = TorchCompiler(fold=True, optimize=True)
    optimized_tc: TorchCircuit = optimized_compiler.compile(sc)

    assert all(
        isinstance(l, (TorchKroneckerLayer, TorchDenseLayer)) for l in unoptimized_tc.layers[-4:]
    )
    assert all(isinstance(l, TorchTuckerLayer) for l in optimized_tc.layers[-2:])

    pnames = [
        ("_nodes.0.logits._nodes.0._ptensor", "_nodes.0.logits._nodes.0._ptensor"),
        ("_nodes.1.logits._nodes.0._ptensor", "_nodes.1.logits._nodes.0._ptensor"),
        ("_nodes.2.weight._nodes.0._ptensor", "_nodes.2.weight._nodes.0._ptensor"),
        ("_nodes.4.weight._nodes.0._ptensor", "_nodes.3.weight._nodes.0._ptensor"),
        ("_nodes.6.weight._nodes.0._ptensor", "_nodes.4.weight._nodes.0._ptensor"),
    ]

    print(unoptimized_tc)
    print()
    print(optimized_tc)
    print()

    for pname, ptensor in unoptimized_tc.named_parameters():
        print(pname, ptensor.shape)
    print()
    for pname, ptensor in optimized_tc.named_parameters():
        print(pname, ptensor.shape)

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc.load_state_dict(
            {optimized_pname: unoptimized_tc.state_dict()[unoptimized_pname]}, strict=False
        )

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)

    unoptimized_scores = unoptimized_tc(worlds)
    assert unoptimized_scores.shape == (2**num_variables, 1, 1)

    optimized_scores = optimized_tc(worlds)
    assert optimized_scores.shape == (2**num_variables, 1, 1)

    assert allclose(unoptimized_scores, optimized_scores)


def test_optimize_candecomp():
    num_variables = 6
    sc = build_simple_pc(num_variables, 3, 2)

    unoptimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=False)
    unoptimized_tc: TorchCircuit = unoptimized_compiler.compile(sc)

    optimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=True)
    optimized_tc: TorchCircuit = optimized_compiler.compile(sc)

    assert all(
        isinstance(l, (TorchHadamardLayer, TorchDenseLayer)) for l in unoptimized_tc.layers[-4:]
    )
    assert all(isinstance(l, TorchCPTLayer) for l in optimized_tc.layers[-2:])

    pnames = [
        ("_nodes.0.logits._nodes.0._ptensor", "_nodes.0.logits._nodes.0._ptensor"),
        ("_nodes.1.logits._nodes.0._ptensor", "_nodes.1.logits._nodes.0._ptensor"),
        ("_nodes.2.weight._nodes.0._ptensor", "_nodes.2.weight._nodes.0._ptensor"),
        ("_nodes.4.weight._nodes.0._ptensor", "_nodes.3.weight._nodes.0._ptensor"),
        ("_nodes.6.weight._nodes.0._ptensor", "_nodes.4.weight._nodes.0._ptensor"),
    ]

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc.load_state_dict(
            {optimized_pname: unoptimized_tc.state_dict()[unoptimized_pname]}, strict=False
        )

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)

    unoptimized_scores = unoptimized_tc(worlds)
    assert unoptimized_scores.shape == (2**num_variables, 1, 1)

    optimized_scores = optimized_tc(worlds)
    assert optimized_scores.shape == (2**num_variables, 1, 1)

    assert allclose(unoptimized_scores, optimized_scores)


def test_optimize_dense_tensordot():
    num_variables = 6
    sc1 = build_simple_pc(num_variables, 3, 2)
    sc2 = build_simple_pc(num_variables, 4, 3)
    sc = SF.multiply(sc1, sc2)

    unoptimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=False)
    unoptimized_tc: TorchCircuit = unoptimized_compiler.compile(sc)
    unoptimized_tc1 = unoptimized_compiler.get_compiled_circuit(sc1)
    unoptimized_tc2 = unoptimized_compiler.get_compiled_circuit(sc2)

    optimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=True)
    optimized_tc: TorchCircuit = optimized_compiler.compile(sc)
    optimized_tc1 = optimized_compiler.get_compiled_circuit(sc1)
    optimized_tc2 = optimized_compiler.get_compiled_circuit(sc2)

    assert all(
        isinstance(l, (TorchCategoricalLayer, TorchHadamardLayer, TorchDenseLayer))
        for l in unoptimized_tc.layers
    )
    assert all(
        isinstance(l, (TorchCategoricalLayer, TorchHadamardLayer, TorchTensorDotLayer))
        for l in optimized_tc.layers
    )

    pnames = [
        ("_nodes.0.logits._nodes.0._ptensor", "_nodes.0.logits._nodes.0._ptensor"),
        ("_nodes.1.logits._nodes.0._ptensor", "_nodes.1.logits._nodes.0._ptensor"),
        ("_nodes.2.weight._nodes.0._ptensor", "_nodes.2.weight._nodes.0._ptensor"),
        ("_nodes.4.weight._nodes.0._ptensor", "_nodes.3.weight._nodes.0._ptensor"),
        ("_nodes.6.weight._nodes.0._ptensor", "_nodes.4.weight._nodes.0._ptensor"),
    ]

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc1.load_state_dict(
            {optimized_pname: unoptimized_tc1.state_dict()[unoptimized_pname]}, strict=False
        )

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc2.load_state_dict(
            {optimized_pname: unoptimized_tc2.state_dict()[unoptimized_pname]}, strict=False
        )

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)

    unoptimized_scores = unoptimized_tc(worlds)
    assert unoptimized_scores.shape == (2**num_variables, 1, 1)

    optimized_scores = optimized_tc(worlds)
    assert optimized_scores.shape == (2**num_variables, 1, 1)

    assert allclose(unoptimized_scores, optimized_scores)
    assert allclose(unoptimized_scores, unoptimized_tc1(worlds) + unoptimized_tc2(worlds))
    assert allclose(optimized_scores, optimized_tc1(worlds) + optimized_tc2(worlds))
    assert allclose(unoptimized_scores, optimized_tc1(worlds) + optimized_tc2(worlds))


def test_optimize_tensordot_squaring():
    num_variables = 12
    sci = build_simple_pc(num_variables, 3, 2)
    sc = SF.multiply(sci, sci)

    unoptimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=False)
    unoptimized_tc: TorchCircuit = unoptimized_compiler.compile(sc)
    unoptimized_tci = unoptimized_compiler.get_compiled_circuit(sci)

    optimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=True)
    optimized_tc: TorchCircuit = optimized_compiler.compile(sc)
    optimized_tci = optimized_compiler.get_compiled_circuit(sci)

    assert all(
        isinstance(l, (TorchCategoricalLayer, TorchHadamardLayer, TorchDenseLayer))
        for l in unoptimized_tc.layers
    )
    assert all(
        isinstance(l, (TorchCategoricalLayer, TorchHadamardLayer, TorchTensorDotLayer))
        for l in optimized_tc.layers
    )

    pnames = [
        ("_nodes.0.logits._nodes.0._ptensor", "_nodes.0.logits._nodes.0._ptensor"),
        ("_nodes.1.logits._nodes.0._ptensor", "_nodes.1.logits._nodes.0._ptensor"),
        ("_nodes.2.weight._nodes.0._ptensor", "_nodes.2.weight._nodes.0._ptensor"),
        ("_nodes.4.weight._nodes.0._ptensor", "_nodes.3.weight._nodes.0._ptensor"),
        ("_nodes.6.weight._nodes.0._ptensor", "_nodes.4.weight._nodes.0._ptensor"),
        ("_nodes.8.weight._nodes.0._ptensor", "_nodes.5.weight._nodes.0._ptensor"),
    ]

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tci.load_state_dict(
            {optimized_pname: unoptimized_tci.state_dict()[unoptimized_pname]}, strict=False
        )

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)

    unoptimized_scores = unoptimized_tc(worlds)
    assert unoptimized_scores.shape == (2**num_variables, 1, 1)

    optimized_scores = optimized_tc(worlds)
    assert optimized_scores.shape == (2**num_variables, 1, 1)

    assert allclose(unoptimized_scores, optimized_scores)
    assert allclose(unoptimized_scores, 2.0 * unoptimized_tci(worlds))
    assert allclose(optimized_scores, 2.0 * optimized_tci(worlds))
    assert allclose(unoptimized_scores, 2.0 * optimized_tci(worlds))


def test_optimize_tensordot_tensordot():
    num_variables = 6
    sc1 = build_simple_pc(num_variables, 3, 2)
    sc2 = build_simple_pc(num_variables, 4, 3)
    sc3 = build_simple_pc(num_variables, 5, 4)
    sc = SF.multiply(sc3, SF.multiply(sc1, sc2))

    unoptimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=False)
    unoptimized_tc: TorchCircuit = unoptimized_compiler.compile(sc)
    unoptimized_tc1 = unoptimized_compiler.get_compiled_circuit(sc1)
    unoptimized_tc2 = unoptimized_compiler.get_compiled_circuit(sc2)
    unoptimized_tc3 = unoptimized_compiler.get_compiled_circuit(sc3)

    optimized_compiler = TorchCompiler(fold=True, semiring="lse-sum", optimize=True)
    optimized_tc: TorchCircuit = optimized_compiler.compile(sc)
    optimized_tc1 = optimized_compiler.get_compiled_circuit(sc1)
    optimized_tc2 = optimized_compiler.get_compiled_circuit(sc2)
    optimized_tc3 = optimized_compiler.get_compiled_circuit(sc3)

    assert all(
        isinstance(l, (TorchCategoricalLayer, TorchHadamardLayer, TorchDenseLayer))
        for l in unoptimized_tc.layers
    )
    assert all(
        isinstance(l, (TorchCategoricalLayer, TorchHadamardLayer, TorchTensorDotLayer))
        for l in optimized_tc.layers
    )

    pnames = [
        ("_nodes.0.logits._nodes.0._ptensor", "_nodes.0.logits._nodes.0._ptensor"),
        ("_nodes.1.logits._nodes.0._ptensor", "_nodes.1.logits._nodes.0._ptensor"),
        ("_nodes.2.weight._nodes.0._ptensor", "_nodes.2.weight._nodes.0._ptensor"),
        ("_nodes.4.weight._nodes.0._ptensor", "_nodes.3.weight._nodes.0._ptensor"),
        ("_nodes.6.weight._nodes.0._ptensor", "_nodes.4.weight._nodes.0._ptensor"),
    ]

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc1.load_state_dict(
            {optimized_pname: unoptimized_tc1.state_dict()[unoptimized_pname]}, strict=False
        )

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc2.load_state_dict(
            {optimized_pname: unoptimized_tc2.state_dict()[unoptimized_pname]}, strict=False
        )

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc3.load_state_dict(
            {optimized_pname: unoptimized_tc3.state_dict()[unoptimized_pname]}, strict=False
        )

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)

    unoptimized_scores = unoptimized_tc(worlds)
    assert unoptimized_scores.shape == (2**num_variables, 1, 1)

    optimized_scores = optimized_tc(worlds)
    assert optimized_scores.shape == (2**num_variables, 1, 1)

    assert allclose(unoptimized_scores, optimized_scores)
    assert allclose(
        unoptimized_scores,
        unoptimized_tc1(worlds) + unoptimized_tc2(worlds) + unoptimized_tc3(worlds),
    )
    assert allclose(
        optimized_scores, optimized_tc1(worlds) + optimized_tc2(worlds) + optimized_tc3(worlds)
    )
    assert allclose(
        unoptimized_scores, optimized_tc1(worlds) + optimized_tc2(worlds) + optimized_tc3(worlds)
    )
