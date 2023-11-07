import functools
from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.region_graph import RegionNode
from cirkit.reparams.leaf import ReparamSigmoid
from cirkit.utils.type_aliases import ReparamFactory

from .exp_family import ExpFamilyLayer

# TODO: rework docstrings


class BinomialLayer(ExpFamilyLayer):
    """Implementation of Binomial distribution."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        nodes: List[RegionNode],
        num_channels: int,
        num_units: int,
        *,
        n: int,
        reparam: ReparamFactory = functools.partial(
            ReparamSigmoid, temperature=10  # type: ignore[misc]
        ),
    ):
        """Init class.

        Args:
            nodes (List[RegionNode]): Passed to super.
            num_channels (int): Number of dims.
            num_units (int): The number of units.
            n (int): n for binomial.
            reparam (int): reparam.
        """
        super().__init__(
            nodes,
            num_channels,
            num_units,
            num_stats=num_channels,
            reparam=reparam,  # TODO: meaning of param/natural_param changed. check if correct
        )
        self.n = n

    def natural_params(self, theta: Tensor) -> Tensor:
        """Calculate natural parameters eta from parameters theta.

        Args:
            theta (Tensor): The parameters theta, shape (D, K, P, S).

        Returns:
            Tensor: The natural parameters eta, shape (D, K, P, S).
        """
        # TODO: torch __rsub__ issue
        return torch.log(theta) - torch.log(1 - theta)  # type: ignore[misc]

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (B, D, S).
        """
        # TODO: confirm dtype compatibility for long/float input and output
        return x  # shape (B, D, S=C)

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (B, D).
        """
        # h(x)=C(n,x)=n!/x!(n-x)!, log(n!)=l[og]gamma(n+1)
        log_h = (
            torch.lgamma(torch.tensor(self.n + 1).to(x))
            - torch.lgamma(x + 1)
            - torch.lgamma(self.n - x + 1)  # type: ignore[misc]  # TODO: torch __rsub__ issue
        )
        return log_h.sum(dim=-1)

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (D, K, P, S).

        Returns:
            Tensor: The log partition function A, shape (D, K, P).
        """
        # TODO: I doubt if this correct, need to check both n==1 and n>1, S=C>1
        # TODO: issue with pylint on torch?
        return self.n * F.softplus(eta).sum(dim=-1)  # pylint: disable=not-callable
