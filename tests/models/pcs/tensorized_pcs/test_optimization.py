import itertools
from typing import Type

import numpy as np
import pytest
import torch
from torch import Tensor

from cirkit.layers import SumLayer
from cirkit.layers.sum_product import (
    BaseCPLayer,
    CollapsedCPLayer,
    SharedCPLayer,
    UncollapsedCPLayer,
)
from cirkit.models import TensorizedPC
from cirkit.models.functional import integrate
from cirkit.reparams.leaf import ReparamExp, ReparamSoftmax
from cirkit.utils import RandomCtx
from tests.models.pcs.tensorized_pcs.test_utils import get_pc_2x2_dense, get_pc_5_sparse


def _optimization_steps(pc: TensorizedPC, data: Tensor, num_steps: int = 5) -> None:
    # Optimize a loss a bit
    opt = torch.optim.SGD(pc.parameters(), lr=0.1)
    pc_pf = integrate(pc)
    loss = np.inf
    for _ in range(num_steps):
        log_z = pc_pf(data)
        log_scores = pc(data)
        lls = log_scores - log_z
        next_loss = -lls.mean()
        assert torch.isfinite(next_loss)
        next_loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        opt.zero_grad()
        assert next_loss.item() < loss
        loss = next_loss.item()


def _check_parameters_sanity(pc: TensorizedPC) -> None:
    # Sanity check on the parameters, by taking into account the folding mask
    for layer in pc.inner_layers:
        if isinstance(layer, SumLayer):
            params = layer.params
            if layer.fold_mask is not None:
                # TODO: fold_mask issue
                fold_mask = layer.fold_mask.view(
                    layer.fold_mask.shape + (1,) * (len(params.shape) - layer.fold_mask.ndim)
                )
                non_masked_params = torch.where(fold_mask.bool(), params(), torch.zeros(()))
                assert torch.all(torch.isfinite(non_masked_params))
            else:
                assert torch.all(torch.isfinite(params()))
        elif isinstance(layer, BaseCPLayer):
            assert layer.params_in is not None
            params = layer.params_in
            if layer.fold_mask is not None:
                fold_mask = layer.fold_mask.view(
                    layer.fold_mask.shape + (1,) * (len(params.shape) - layer.fold_mask.ndim)
                )
                mask = fold_mask[0].bool() if isinstance(layer, SharedCPLayer) else fold_mask.bool()
                non_masked_params = torch.where(mask, params(), torch.zeros(()))
                assert torch.all(torch.isfinite(non_masked_params))
            else:
                assert torch.all(torch.isfinite(params()))
        else:
            assert False, type(layer)


@torch.set_grad_enabled(True)
@pytest.mark.parametrize(
    "normalized,layer_cls",
    list(itertools.product([False, True], [CollapsedCPLayer, UncollapsedCPLayer, SharedCPLayer])),
)
@RandomCtx(42)
def test_pc_dense_backprop(normalized: bool, layer_cls: Type[BaseCPLayer]) -> None:
    reparam = ReparamSoftmax if normalized else ReparamExp
    pc = get_pc_2x2_dense(reparam, layer_cls, num_units=2)
    data = torch.tensor(list(itertools.product([0, 1], repeat=4))).unsqueeze(  # type: ignore[misc]
        dim=-1
    )
    _optimization_steps(pc, data)
    _check_parameters_sanity(pc)


@torch.set_grad_enabled(True)
@pytest.mark.parametrize(
    "normalized,layer_cls",
    list(itertools.product([False, True], [CollapsedCPLayer, UncollapsedCPLayer, SharedCPLayer])),
)
@RandomCtx(42)
def test_pc_sparse_backprop(normalized: bool, layer_cls: Type[BaseCPLayer]) -> None:
    reparam = ReparamSoftmax if normalized else ReparamExp
    pc = get_pc_5_sparse(reparam, layer_cls, num_units=2)
    data = torch.tensor(list(itertools.product([0, 1], repeat=5))).unsqueeze(  # type: ignore[misc]
        dim=-1
    )
    _optimization_steps(pc, data)
    _check_parameters_sanity(pc)
