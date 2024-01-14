import functools
import itertools

import pytest
import torch

from cirkit.layers.sum_product import TuckerLayer
from cirkit.utils import RandomCtx
from cirkit.new.reparams.unary import UnaryReparam
from cirkit.new.utils.type_aliases import ReparamFactory
from tests import floats


@pytest.mark.parametrize(
    "num_input_units,num_output_units,num_folds,arity,reparam_name",
    list(itertools.product([1, 2], [1, 3], [1, 4], [3, 3], ["unary"])),
)
@RandomCtx(42)
def test_tucker_layer(
    num_input_units: int, num_output_units: int, num_folds: int, arity: int, reparam_name: str
) -> None:
    reparam: ReparamFactory
    if reparam_name == "unary":
        reparam = UnaryReparam(func=torch.tanh)
    else:
        assert False

    if arity != 2:
        with pytest.raises(NotImplementedError):
            TuckerLayer(
                num_input_units=num_input_units,
                num_output_units=num_output_units,
                arity=arity,  # type: ignore[arg-type]
                num_folds=num_folds,
                reparam=reparam,
            )
        return
    layer = TuckerLayer(
        num_input_units=num_input_units,
        num_output_units=num_output_units,
        # arity=2,
        num_folds=num_folds,
        reparam=reparam,
    )

    batch_size = 12
    x = torch.randn(num_folds, arity, num_input_units, batch_size)  
    output = layer(x) 
    assert not floats.allclose(output, 0.0)
    assert output.shape == (num_folds, num_output_units, batch_size)

