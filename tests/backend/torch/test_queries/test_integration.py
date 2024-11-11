import itertools

import pytest
import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.queries import IntegrateQuery
from cirkit.utils.scope import Scope
from tests.floats import allclose
from tests.symbolic.test_utils import build_monotonic_structured_categorical_cpt_pc


@pytest.mark.parametrize(
    "semiring,fold,optimize",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True]),
)
def test_query_marginalize_monotonic_pc_categorical(semiring: str, fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    # The following function computes a circuit where we have computed the
    # partition function and a marginal by hand.
    sc, gt_outputs, gt_partition_func = build_monotonic_structured_categorical_cpt_pc(
        return_ground_truth=True
    )

    mar_sc = SF.integrate(sc, scope=Scope([4]))
    mar_tc: TorchCircuit = compiler.compile(mar_sc)
    assert isinstance(mar_tc, TorchCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

    mar_worlds = torch.cat(
        [
            torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables - 1))).unsqueeze(
                dim=-2
            ),
            torch.zeros(2 ** (tc.num_variables - 1), dtype=torch.int64)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        ],
        dim=2,
    )
    mar_scores1 = mar_tc(mar_worlds)
    mar_query = IntegrateQuery(tc)
    mar_scores2 = mar_query(mar_worlds, integrate_vars=Scope([4]))
    assert mar_scores1.shape == mar_scores2.shape
    assert allclose(mar_scores1, mar_scores2)


@pytest.mark.parametrize(
    "semiring,fold,optimize",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True]),
)
def test_batch_query_marginalize_monotonic_pc_categorical(
    semiring: str, fold: bool, optimize: bool
):
    # Check using a mask with batching works
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    # The following function computes a circuit where we have computed the
    # partition function and a marginal by hand.
    sc, gt_outputs, gt_partition_func = build_monotonic_structured_categorical_cpt_pc(
        return_ground_truth=True
    )

    tc: TorchCircuit = compiler.compile(sc)

    # The marginal has been computed for (1, 0, 1, 1, None) -- so marginalising var 4.
    inputs = torch.tensor([[[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]]], dtype=torch.int64).view(2, 1, 5)

    mar_query = IntegrateQuery(tc)
    # Create two masks, one is marginalising out everything
    # and another is marginalising out only the last variable
    mask = [Scope([0, 1, 2, 3, 4]), Scope([4])]
    # The first score should be partition function, as we marginalised out all vars.
    # The second score, should be our precomputed marginal.
    mar_scores = mar_query(inputs, integrate_vars=mask)

    if semiring == "sum-product":
        assert torch.isclose(mar_scores[0], torch.tensor(gt_partition_func))
        assert torch.isclose(mar_scores[1], torch.tensor(gt_outputs["mar"][(1, 0, 1, 1, None)]))
    elif semiring == "lse-sum":
        mar_scores = torch.exp(mar_scores)
        assert torch.isclose(mar_scores[0], torch.tensor(gt_partition_func))
        assert torch.isclose(mar_scores[1], torch.tensor(gt_outputs["mar"][(1, 0, 1, 1, None)]))
    else:
        raise ValueError('Unexpected semiring: "%s"' % semiring)


@pytest.mark.parametrize(
    "semiring,fold,optimize",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True]),
)
def test_batch_broadcast_query_marginalize_monotonic_pc_categorical(
    semiring: str, fold: bool, optimize: bool
):
    # Check that passing a single mask results in broadcasting
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    # The following function computes a circuit where we have computed the
    # partition function and a marginal by hand.
    sc, gt_outputs, gt_partition_func = build_monotonic_structured_categorical_cpt_pc(
        return_ground_truth=True
    )

    tc: TorchCircuit = compiler.compile(sc)

    # The marginal has been computed for (1, 0, 1, 1, None) -- so marginalising var 4.
    inputs = torch.tensor([[[1, 0, 1, 1, 0], [1, 0, 1, 1, 1]]], dtype=torch.int64).view(2, 1, 5)

    mar_query = IntegrateQuery(tc)
    # Create a single mask - this should be broadcast along the batch dim.
    mask = Scope([4])
    # The first score should be partition function, as we marginalised out all vars.
    # The second score, should be our precomputed marginal.
    mar_scores = mar_query(inputs, integrate_vars=mask)

    if semiring == "sum-product":
        assert torch.isclose(mar_scores[0], torch.tensor(gt_outputs["mar"][(1, 0, 1, 1, None)]))
        assert torch.isclose(mar_scores[1], torch.tensor(gt_outputs["mar"][(1, 0, 1, 1, None)]))
    elif semiring == "lse-sum":
        mar_scores = torch.exp(mar_scores)
        assert torch.isclose(mar_scores[0], torch.tensor(gt_outputs["mar"][(1, 0, 1, 1, None)]))
        assert torch.isclose(mar_scores[1], torch.tensor(gt_outputs["mar"][(1, 0, 1, 1, None)]))
    else:
        raise ValueError('Unexpected semiring: "%s"' % semiring)


def test_batch_fails_on_out_of_scope(semiring="sum-product", fold=True, optimize=True):
    # Check that passing a single mask results in broadcasting
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    # The following function computes a circuit where we have computed the
    # partition function and a marginal by hand.
    sc, gt_outputs, gt_partition_func = build_monotonic_structured_categorical_cpt_pc(
        return_ground_truth=True
    )

    tc: TorchCircuit = compiler.compile(sc)

    # The marginal has been computed for (1, 0, 1, 1, None) -- so marginalising var 4.
    inputs = torch.tensor([[[1, 0, 1, 1, 0], [1, 0, 1, 1, 1]]], dtype=torch.int64).view(2, 1, 5)

    mar_query = IntegrateQuery(tc)
    # Scope 5 does not exist so this should error
    mask = [Scope([0]), Scope([5])]
    # The first score should be partition function, as we marginalised out all vars.
    # The second score, should be our precomputed marginal.
    with pytest.raises(ValueError, match="not in scope:.*?5"):
        mar_scores = mar_query(inputs, integrate_vars=mask)
