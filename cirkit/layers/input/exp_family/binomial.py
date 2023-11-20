from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory

from .exp_family import ExpFamilyLayer


class BinomialLayer(ExpFamilyLayer):
    """The binomial distribution layer."""

    def __init__(  # pylint: disable=too-many-arguments
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
        n: int,
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
            n (int): The parameter n for bimonial distribution.
        """
        assert n > 0, "The parameter n for bimonial distribution must be positive."
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
            num_suff_stats=num_channels,
        )
        self.n = n

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, D, S).
        """
        # TODO: confirm dtype compatibility for long/float input and output
        return x  # shape (*B, D, S=C)

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (*B, D).
        """
        # h(x)=C(n,x)=n!/x!(n-x)!, log(n!)=l[og]gamma(n+1)
        log_h = (
            torch.lgamma(torch.tensor(self.n + 1).to(x))
            - torch.lgamma(x + 1)
            - torch.lgamma(self.n - x + 1)  # type: ignore[misc]  # TODO: torch __rsub__ issue
        )  # shape (*B, D, C)
        return log_h.sum(dim=-1)  # shape (*B, D)

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (D, K, P, S).

        Returns:
            Tensor: The log partition function A, shape (D, K, P).
        """
        # TODO: I doubt if this correct, need to check both n==1 and n>1, S=C>1
        # TODO: issue with pylint on torch?
        # pylint: disable-next=not-callable
        return self.n * F.softplus(eta).sum(dim=-1)  # shape (D, K, P, S) -> (D, K, P)

    @property
    def p(self) -> Tensor:
        """The parameter p for bimonial distribution, shape (D, K, P, C)."""
        return torch.sigmoid(self.params())  # shape (D, K, P, C=S)
