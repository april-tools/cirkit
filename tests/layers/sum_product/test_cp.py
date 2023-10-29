import functools
import itertools
import pytest
import torch

from cirkit.layers.sum_product import CPLayer
from cirkit.utils import RandomCtx
from cirkit.utils.reparams import reparam_positive, reparam_softmax


@pytest.mark.parametrize(
    "num_input_units,num_output_units,num_folds,arity,uncollapsed,reparam_name",
    list(
        itertools.product(
            [1, 2], [1, 3], [1, 4], [2, 3], [False, True], ['positive', 'softmax']
        )
    ),
)
@RandomCtx(42)
def test_cp_layer(
        num_input_units: int,
        num_output_units: int,
        num_folds: int,
        arity: int,
        uncollapsed: bool,
        reparam_name: str
) -> None:
    if reparam_name == "softmax":
        reparam = functools.partial(reparam_softmax, dim=-2)
    elif reparam_name == "positive":
        reparam = functools.partial(reparam_positive, eps=1e-7)
    else:
        assert False

    rank = 5
    layer = CPLayer(
        num_input_units, num_output_units,
        arity=arity, num_folds=num_folds, reparam=reparam, uncollapsed=uncollapsed, rank=rank)  # type: ignore[misc]

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
