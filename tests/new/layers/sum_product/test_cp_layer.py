

import pytest
import torch

from cirkit.new.utils.type_aliases import ReparamFactory
from cirkit.utils import RandomCtx
from cirkit.new.layers.inner.sum_product.cp import CPLayer 
from cirkit.new.reparams.unary import UnaryReparam
import functools
import itertools


@pytest.mark.parametrize(
    "num_input_units,num_output_units,arity,reparam_name",
    list(itertools.product([1, 2], [1, 3], [1, 1],["unary"])),
)
@RandomCtx(42)
# pylint: disable-next=too-many-arguments
def test_cp_layer(
    num_input_units: int,
    num_output_units: int,
    arity: int,
    reparam_name: str,
) -> None:
    reparam: ReparamFactory
    if reparam_name == "unary":
        reparam = UnaryReparam(func=torch.tanh)

    layer = CPLayer(
        num_input_units=num_input_units,
        num_output_units=num_output_units,
        arity=arity,
        reparam=reparam,
    )

    batch_size = 2 
    x = torch.randn(arity, num_input_units, batch_size)  
    output = layer(x)
    assert output.shape == (num_input_units, num_output_units,)