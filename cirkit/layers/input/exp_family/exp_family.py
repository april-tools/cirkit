from abc import abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor, nn

from cirkit.layers.input import InputLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.reparams.reparam import Reparameterizaion
from cirkit.utils.type_aliases import ReparamFactory


class ExpFamilyLayer(InputLayer):
    """The abstract base for Exponential Family distribution input layers.

    Exponential Family: f(x|theta) = exp(eta(theta) dot T(x) - log_h(x) + A(eta)).
    Ref: https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions

    However here we don't parameterize theta but directly use eta instead. It's the duty of \
    subclasses to define properties to provide parameter theta (or different components of theta) \
    based on its implementation.
    """

    params: Reparameterizaion
    """The reparameterizaion that gives the natural parameters eta, shape (D, K, P, S)."""

    def __init__(  # type: ignore[misc]  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[0] = 0,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamIdentity,
        num_suff_stats: int = -1,
        **_: Any,
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
            num_folds (Literal[0], optional): The number of folds. Should not be provided and will \
                be calculated as num_vars*num_replicas. Defaults to 0.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
            num_suff_stats (int, optional): The number of sufficient statistics, as required by \
                each implementation. The default value is not valid, but only a hint for \
                not-required as it does not appear in subclasses. Defaults to -1.
        """
        assert num_suff_stats > 0, "The number of sufficient statistics must be positive."
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
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, D, S).
        """

    @abstractmethod
    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (*B, D).
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
            x (Tensor): The input to this layer, shape (*B, D, C).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        eta = self.params()  # shape (D, K, P, S)
        suff_stats = self.sufficient_stats(x)  # shape (*B, D, S)
        log_h = self.log_base_measure(x)  # shape (*B, D)
        log_part = self.log_partition(eta)  # shape (D, K, P)
        return (
            torch.einsum("dkps,...ds->...dkp", eta, suff_stats)  # shape (*B, D, K, P)
            + log_h.unsqueeze(dim=-1).unsqueeze(dim=-1)  # shape (*B, D, 1, 1)
            - log_part  # shape (*1, D, K, P), 1s automatically prepended
        )  # shape (*B, D, K, P)

    def integrate(self) -> Tensor:
        """Return the integation, which is a zero tensor for this layer (in log-space).

        Returns:
            Tensor: A zero tensor of shape (1, num_vars, num_units, num_replicas).
        """
        # TODO: return an expanded zeros?
        # TODO: output shape should be (*B, D, K, P)
        return torch.zeros(
            size=(1, self.num_vars, self.num_output_units, self.num_replicas),
            requires_grad=False,
            device=self.params().device,  # TODO: this is not good
        )

    # TODO: see 241d46a43f59c1df23b5136a45b5f18b9f116671 for backtrack
