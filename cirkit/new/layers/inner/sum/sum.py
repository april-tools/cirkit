import functools

import torch
from torch import nn

from cirkit.new.layers.inner.inner import InnerLayer
from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import KroneckerReparam, Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class SumLayer(InnerLayer):
    """The abstract base class for sum layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. Although sum layers typically have
    #       parameters, we still allow it to be optional for flexibility.

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        for child in self.children():
            if isinstance(child, Reparameterization):
                child.initialize(functools.partial(nn.init.uniform_, a=0.01, b=0.99))

    @classmethod
    def get_product(
        cls, left_symb_cfg: SymbLayerCfg[Layer], right_symb_cfg: SymbLayerCfg[Layer]
    ) -> SymbLayerCfg[Layer]:
        """Get the symbolic config to construct the product of this layer and the other layer.

        Subclasses of SumLayer only allows product with the same class. However, the signature \
        typing is not narrowed down, and wrong arg type will not be captured by static checkers \
        but only during runtime.

        For any layer with sum weights, the product with the same class is still the class, with:
            param = param_a ⊗ param_b,
        unless a specific implementation need to change this behaviour.

        Args:
            left_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the left operand.
            right_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the right operand.

        Returns:
            SymbLayerCfg[Layer]: The symbolic config for the product.
        """
        assert (
            issubclass(left_symb_cfg.layer_cls, cls)
            and left_symb_cfg.layer_cls == right_symb_cfg.layer_cls
        ), "Both inputs of SumLayer.get_product must be of self class."

        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=left_symb_cfg.layer_cls,
            layer_kwargs=left_symb_cfg.layer_kwargs,  # type: ignore[misc]
            reparam=KroneckerReparam(left_symb_cfg.reparam, right_symb_cfg.reparam),
        )
