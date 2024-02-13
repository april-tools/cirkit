import functools
from typing_extensions import Self  # FUTURE: in typing from 3.11

import torch
from torch import nn

from cirkit.new.layers.inner.inner import InnerLayer
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
        cls,
        self_symb_cfg: SymbLayerCfg[Self],
        other_symb_cfg: SymbLayerCfg[Self],
    ) -> SymbLayerCfg[Self]:
        """Get the symbolic config to construct the product of this sum layer with the other sum \
        layer.

        The two inner layers for product must be of the same layer class. This means that the \
        product must be performed between two circuits with the same region graph.

        This will construct a new layer config with param = param_self ⊗ param_other. This \
        applies to all sum layers and sum-product layers. See each layer for details.

        Args:
            self_symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            other_symb_cfg (SymbLayerCfg[Self]): The symbolic config for the other layer.

        Returns:
            SymbLayerCfg[Self]: The symbolic config for the product of the two sum layers.
        """
        assert (
            self_symb_cfg.layer_cls == other_symb_cfg.layer_cls
        ), "Both layers must be of the same class."

        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=self_symb_cfg.layer_cls,
            layer_kwargs=self_symb_cfg.layer_kwargs,  # type: ignore[misc]
            reparam=KroneckerReparam(self_symb_cfg.reparam, other_symb_cfg.reparam),
        )
