import math

import torch
from torch import Tensor

from cirkit.backend.torch.layers.input.ef import TorchExpFamilyLayer
from cirkit.backend.torch.params import TorchGaussianMeanProductParameter


class ProdEFLayer(TorchExpFamilyLayer):
    """The product for Exponential Family distribution layers.

    Exponential Family dist:
        p(x|η) = exp(η · T(x) + log_h(x) - A(η)).

    Product:
        p_1*p_2(x|η_1,η_2) = exp(η · T(x) + log_h(x) - A(η)),
    where:
        - η = concat(η_1, η_2),
        - T(x) = concat(T_1(x), T_2(x)),
        - log_h(x) = log_h_1(x) + log_h_2(x),
        - A(η) = A_1(η_1) + A_2(η_2).

    However the A here is not the log partition anymore, so get_integral should not be 1s.
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: TorchGaussianMeanProductParameter,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (TorchGaussianMeanProductParameter): The reparameterization for layer parameters.
            ef1_cfg (SymbLayerCfg[ExpFamilyLayer]): The config of the left ExpFamilyLayer for \
                product, should include a reference to a concretized SymbL for EF.
            ef2_cfg (SymbLayerCfg[ExpFamilyLayer]): The config of the right ExpFamilyLayer for \
                product, should include a reference to a concretized SymbL for EF.
        """
        assert isinstance(
            reparam, TorchGaussianMeanProductParameter
        ), "Must use a EFProductReparam for ProdEFLayer."

        self.suff_split_point = math.prod(ef1.suff_stats_shape)
        self.suff_stats_shape = (self.suff_split_point + math.prod(ef2.suff_stats_shape),)

        # NOTE: We need a reparam to be provided in SymbLayerCfg and registered for EFLayer.
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        # We need suff_stats_shape before __init__, but submodule can only be registered after.
        self.ef1 = ef1
        self.ef2 = ef2

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The sufficient statistics T, shape (H, *B, *S).
        """
        # shape (H, *B, *S_1), (H, *B, *S_2) -> (H, *B, flatten(S_1)+flatten(S_2)).
        return torch.cat(
            (
                self.ef1.sufficient_stats(x).flatten(start_dim=-len(self.ef1.suff_stats_shape)),
                self.ef2.sufficient_stats(x).flatten(start_dim=-len(self.ef2.suff_stats_shape)),
            ),
            dim=-1,
        )

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The natural parameters eta, shape (H, *B).
        """
        return self.ef1.log_base_measure(x) + self.ef2.log_base_measure(x)

    def log_partition(self, eta: Tensor, *, eta_normed: bool = False) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, Ko, *S).
            eta_normed (bool, optional): Ignored. This layer uses a special reparam for eta. \
                Defaults to False.

        Returns:
            Tensor: The log partition function A, shape (H, Ko).
        """
        # TODO: x.unflatten is not typed
        eta1 = torch.unflatten(
            eta[..., : self.suff_split_point], dim=-1, sizes=self.ef1.suff_stats_shape
        )
        eta2 = torch.unflatten(
            eta[..., self.suff_split_point :], dim=-1, sizes=self.ef2.suff_stats_shape
        )

        # TODO: eta_normed?
        return self.ef1.log_partition(eta1) + self.ef2.log_partition(eta2)
