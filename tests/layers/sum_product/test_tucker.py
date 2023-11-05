# pylint: disable=missing-function-docstring
import functools
import itertools

import pytest
import torch

from cirkit.layers.sum_product import TuckerLayer
from cirkit.reparams.leaf import ReparamClamp, ReparamSoftmax
from cirkit.utils import RandomCtx
from cirkit.utils.type_aliases import ReparamFactory


@pytest.mark.parametrize(
    "num_input_units,num_output_units,num_folds,arity,reparam_name",
    list(itertools.product([1, 2], [1, 3], [1, 4], [2, 3], ["positive", "softmax"])),
)
@RandomCtx(42)
def test_tucker_layer(
    num_input_units: int, num_output_units: int, num_folds: int, arity: int, reparam_name: str
) -> None:
    reparam: ReparamFactory
    if reparam_name == "softmax":
        reparam = ReparamSoftmax
    elif reparam_name == "positive":
        reparam = functools.partial(ReparamClamp, min=1e-7)  # type: ignore[misc]
    else:
        assert False

    if arity != 2:
        with pytest.raises(NotImplementedError):
            TuckerLayer(
                num_input_units, num_output_units, arity=arity, num_folds=num_folds, reparam=reparam
            )
        return
    layer = TuckerLayer(
        num_input_units, num_output_units, arity=arity, num_folds=num_folds, reparam=reparam
    )

    batch_size = 16
    x = torch.randn(num_folds, arity, num_input_units, batch_size)  # (F, H, K, B)
    output = layer(x)  # (F, J, B)
    assert not torch.allclose(output, torch.zeros(()))
    assert output.shape == torch.Size([num_folds, num_output_units, batch_size])

    if reparam_name == "softmax":
        x = torch.zeros(num_folds, arity, num_input_units, batch_size)  # (F, H, K, B)
        output = layer(x)  # (F, J, B)
        assert output.shape == torch.Size([num_folds, num_output_units, batch_size])
        assert torch.allclose(output, torch.zeros(()), atol=2e-7)
