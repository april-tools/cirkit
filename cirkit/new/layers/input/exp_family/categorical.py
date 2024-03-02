from typing_extensions import Never, Self  # FUTURE: in typing from 3.11

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class CategoricalLayer(ExpFamilyLayer):
    """The Categorical distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Reparameterization,
        num_categories: int,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
            num_categories (int): The number of categories for Categorical distribution.
        """
        assert (
            num_categories > 0
        ), "The number of categories for Categorical distribution must be positive."
        self.num_categories = num_categories
        self.suff_stats_shape = (num_input_units, num_categories)
        # Set self.suff_stats_shape before ExpFamilyLayer.__init__. The reparam will be set in
        # ExpFamilyLayer.__init__ to normalize dim=-1 (cat).
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The sufficient statistics T, shape (H, *B, *S).
        """
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete.
        # TODO: pylint issue? one_hot is only in pyi
        # pylint: disable-next=not-callable
        suff_stats = F.one_hot(x, self.num_categories)  # shape (H, *B, Ki) -> (H, *B, Ki, cat).
        return suff_stats.to(torch.get_default_dtype())  # The suff stats must be float.

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The natural parameters eta, shape (H, *B).
        """
        return x.new_zeros(()).expand(x.shape[:-1])

    def log_partition(self, eta: Tensor, *, eta_normed: bool = False) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, Ko, *S).
            eta_normed (bool, optional): Whether eta is produced by a NormalizedReparam. If True, \
                implementations may save some computation. Defaults to False.

        Returns:
            Tensor: The log partition function A, shape (H, Ko).
        """
        if eta_normed:
            return eta.new_zeros(()).expand(eta.shape[:2])

        # If eta is not normalized, we need this to make sure the output of EF is normalized.
        # shape (H, Ko, Ki, cat) -> (H, Ko, Ki) -> (H, Ko).
        return eta.logsumexp(dim=-1).sum(dim=-1)

    @classmethod
    def get_partial(
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> Never:
        """Get the symbolic config to construct the partial differential w.r.t. the given channel \
        of the given variable in the scope of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            order (int, optional): The order of differentiation. Defaults to 1.
            var_idx (int, optional): The variable to diffrentiate. The idx is counted within this \
                layer's scope but not global variable id. Defaults to 0.
            ch_idx (int, optional): The channel of variable to diffrentiate. Defaults to 0.

        Raises:
            TypeError: When this method is called on CategoricalLayer.
        """
        raise TypeError("Cannot differentiate over discrete variables.")
