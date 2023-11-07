from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from cirkit.layers.input import InputLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory


class ExpFamilyLayer(InputLayer):
    """The abstract base for Exponential Family distribution input layers.

    Exponential Family: f(x|theta) = exp(eta(theta) dot T(x) - log_h(x) + A(eta)).
    Ref: https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions

    However here we don't parameterize theta but directly use eta instead.
    Subclasses define properties to provide parameter theta based on its implementation.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[-1] = -1,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamIdentity,
        num_suff_stats: int = -1,
    ) -> None:
        """Init class.

        Args:
            num_vars (int): The number of variables of the circuit.
            num_channels (int, optional): The number of channels of each variable. Defaults to 1.
            num_replicas (int, optional): The number of replicas for each variable. Defaults to 1.
            num_input_units (Literal[1], optional): The number of input units, must be 1. \
                Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            num_folds (Literal[-1], optional): The number of folds, unused. The number of folds \
                should be num_vars*num_replicas. Defaults to -1.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
            num_suff_stats (int, optional): The number of sufficient statistics, as required by \
                each implementation. Defaults to -1.
        """
        assert num_suff_stats > 0
        super().__init__(
            num_vars=num_vars,
            num_channels=num_channels,
            num_replicas=num_replicas,
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
        )
        self.num_suff_stats = num_suff_stats

        self.params = reparam(
            (self.num_vars, self.num_output_units, self.num_replicas, self.num_suff_stats), dim=-1
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: N(0, 1)."""
        for param in self.parameters():
            nn.init.normal_(param, 0, 1)

    @abstractmethod
    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (B, D, S).
        """

    @abstractmethod
    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (B, D).
        """

    @abstractmethod
    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (D, K, P, S).

        Returns:
            Tensor: The log partition function A, shape (D, K, P).
        """

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (B, D, C).

        Returns:
            Tensor: The output of this layer, shape (B, D, K, P).
        """
        # TODO: this does not work for more than 1 batch dims
        if x.ndim == 2:
            x = x.unsqueeze(dim=-1)

        eta = self.params()  # shape (D, K, P, S)
        suff_stats = self.sufficient_stats(x)  # shape (B, D, S)
        log_h = self.log_base_measure(x)  # shape (B, D)
        log_part = self.log_partition(eta)  # shape (D, K, P)
        return (
            torch.einsum("dkps,bds->bdkp", eta, suff_stats)  # shape (B, D, K, P)
            + log_h.unsqueeze(dim=-1).unsqueeze(dim=-1)  # shape (B, D, 1, 1)
            - log_part.unsqueeze(dim=0)  # shape (1, D, K, P)
        )  # shape (B, D, K, P)

    def integrate(self) -> Tensor:
        """Return the integation, which is a zero tensor for this layer (in log-space).

        Returns:
            Tensor: A zero tensor of shape (1, num_vars, num_units, num_replicas).
        """
        return torch.zeros(
            size=(1, self.num_vars, self.num_output_units, self.num_replicas),
            requires_grad=False,
            device=self.params().device,  # TODO: this is not good
        )

    # TODO: see 241d46a43f59c1df23b5136a45b5f18b9f116671 for backtrack
