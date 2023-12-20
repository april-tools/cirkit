import functools
import itertools

import pytest
import torch

from cirkit.layers.sum_product import CPLayer
from cirkit.reparams.leaf import ReparamClamp, ReparamSoftmax
from cirkit.utils import RandomCtx
from cirkit.utils.type_aliases import ReparamFactory
from tests import floats


@pytest.mark.parametrize(
    "num_input_units,num_output_units,num_folds,arity,uncollapsed,reparam_name",
    list(itertools.product([1, 2], [1, 3], [1, 4], [2, 3], [False, True], ["positive", "softmax"])),
)
@RandomCtx(42)
# pylint: disable-next=too-many-arguments
def test_cp_layer(
    num_input_units: int,
    num_output_units: int,
    num_folds: int,
    arity: int,
    uncollapsed: bool,
    reparam_name: str,
) -> None:
    reparam: ReparamFactory
    if reparam_name == "softmax":
        reparam = ReparamSoftmax
    elif reparam_name == "positive":
        reparam = functools.partial(ReparamClamp, min=1e-7)  # type: ignore[misc]
    else:
        assert False

    rank = 5
    layer = CPLayer(
        num_input_units=num_input_units,
        num_output_units=num_output_units,
        arity=arity,
        num_folds=num_folds,
        reparam=reparam,
        collapsed=not uncollapsed,
        rank=rank,
    )

    batch_size = 16
    x = torch.randn(num_folds, arity, num_input_units, batch_size)  # (F, H, K, B)
    output = layer(x)  # (F, J, B)
    assert not floats.allclose(output, 0.0)
    assert output.shape == (num_folds, num_output_units, batch_size)

    if reparam_name == "softmax":
        x = torch.zeros(num_folds, arity, num_input_units, batch_size)  # (F, H, K, B)
        output = layer(x)  # (F, J, B)
        assert output.shape == (num_folds, num_output_units, batch_size)
        assert floats.allclose(output, 0.0)
