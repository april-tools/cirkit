import functools
import itertools
import pytest
import torch

from cirkit.layers.sum_product import TuckerLayer
from cirkit.utils import RandomCtx
from cirkit.utils.reparams import reparam_positive, reparam_softmax


@pytest.mark.parametrize(
    "num_input_units,num_output_units,num_folds,arity,reparam_name",
    list(
        itertools.product(
            [1, 2], [1, 3], [1, 4], [2, 3], ['positive', 'softmax']
        )
    ),
)
@RandomCtx(42)
def test_tucker_layer(
        num_input_units: int,
        num_output_units: int,
        num_folds: int,
        arity: int,
        reparam_name: str
) -> None:
    if reparam_name == "softmax":
        reparam_func = functools.partial(reparam_softmax, dim=-2)
        reparam = lambda w, _: reparam_func(w.view(w.shape[0], -1, w.shape[3])).view(*w.shape)
    elif reparam_name == "positive":
        reparam = functools.partial(reparam_positive, eps=1e-7)
    else:
        assert False

    if arity != 2:
        with pytest.raises(NotImplementedError):
            TuckerLayer(
                num_input_units, num_output_units,
                arity=arity, num_folds=num_folds, reparam=reparam)  # type: ignore[misc]
        return
    layer = TuckerLayer(
        num_input_units, num_output_units,
        arity=arity, num_folds=num_folds, reparam=reparam)  # type: ignore[misc]

    batch_size = 16
    input = torch.randn(num_folds, arity, num_input_units, batch_size)  # (F, H, K, B)
    output = layer(input)  # (F, J, B)
    assert not torch.allclose(output, torch.zeros(()))
    assert output.shape == torch.Size([num_folds, num_output_units, batch_size])

    if reparam_name == 'softmax':
        input = torch.zeros(num_folds, arity, num_input_units, batch_size)  # (F, H, K, B)
        output = layer(input)  # (F, J, B)
        assert output.shape == torch.Size([num_folds, num_output_units, batch_size])
        assert torch.allclose(output, torch.zeros(()), atol=2e-7)
