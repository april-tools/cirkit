import functools
from typing import Callable

from torch import Tensor, nn

from cirkit.new.layers.inner.inner import InnerLayer
from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import KroneckerReparam
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


class SumLayer(InnerLayer):
    """The abstract base class for sum layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. Although sum layers typically have
    #       parameters, we still allow it to be optional for flexibility.

    @property
    def _default_initializer_(self) -> Callable[[Tensor], Tensor]:
        """The default inplace initializer for the parameters of this layer.

        The sum weights are initialized to U(0.01, 0.99).
        """
        return functools.partial(nn.init.uniform_, a=0.01, b=0.99)

    @classmethod
    def get_product(
        cls, left_symb_cfg: SymbLayerCfg[Layer], right_symb_cfg: SymbLayerCfg[Layer]
    ) -> SymbCfgFactory[Layer]:
        """Get the symbolic config to construct the product of this layer and the other layer.

        Subclasses of SumLayer can only be multiplied with the same class. However, the signature \
        typing is not narrowed down, and wrong arg type will not be captured by static checkers \
        but only during runtime.

        For any layer with sum weights, the product with the same class is still the class, with:
            param = param_a ⊗ param_b,
        unless a specific implementation need to change this behaviour.

        Args:
            left_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the left operand.
            right_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the right operand.

        Returns:
            SymbCfgFactory[Layer]: The symbolic config for the product. NOTE: Implicit to typing, \
                NotImplemented may also be returned, which indicates the reflection should be tried.
        """
        assert (
            issubclass(left_symb_cfg.layer_cls, cls)
            and left_symb_cfg.layer_cls == right_symb_cfg.layer_cls
        ), "Both inputs to SumLayer.get_product must be the same and of self class."

        # SumLayer may also be SumProdLayer, and we use the existence of reparam to determine.
        reparam = (
            KroneckerReparam(left_symb_cfg.reparam, right_symb_cfg.reparam)
            if left_symb_cfg.reparam is not None and right_symb_cfg.reparam is not None
            else None
        )
        # IGNORE: Unavoidable for kwargs.
        return SymbCfgFactory(
            layer_cls=left_symb_cfg.layer_cls,
            layer_kwargs=left_symb_cfg.layer_kwargs,  # type: ignore[misc]
            reparam=reparam,
        )
