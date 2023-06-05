from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch import Tensor, nn


# TODO: name it layer?
class Layer(nn.Module, ABC):
    """Abstract layer class. Specifies functionality every layer in an EiNet should implement."""

    def __init__(self) -> None:
        """Init class."""
        super().__init__()  # TODO: do we need multi-inherit init?
        self.prob: Optional[Tensor] = None  # TODO: why None?

    @abstractmethod
    def default_initializer(self) -> Tensor:
        """Produce suitable initial parameters for the layer.

        :return: initial parameters
        """

    @abstractmethod
    def initialize(self, initializer: Optional[Tensor] = None) -> None:
        """Initialize the layer, e.g. with return value from default_initializer(self).

        :param initializer: 'default', or custom (typically a Tensor)
                            'default' means that the layer simply calls its own \
                                default_initializer(self), in stores
                            the parameters internally.
                            custom (typically a Tensor) means that you pass your own initializer.
        :return: None
        """

    def __call__(self, x: Optional[Tensor] = None) -> None:
        """Invoke the forward.

        Args:
            x (Optional[Tensor], optional): The input. Defaults to None.
        """
        super().__call__(x)

    # TODO: it's not good to return None
    @abstractmethod
    def forward(self, x: Optional[Tensor] = None) -> None:
        """Compute the layer. The result is always a tensor of \
            log-densities of shape (batch_size, num_dist, num_nodes), \
        where num_dist is the vector length (K in the paper) and num_nodes is \
            the number of PC nodes in the layer.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, \
                    self.num_var, self.num_dims).
                  Not all layers use this argument.
        :return: log-density tensor of shape (batch_size, num_dist, num_nodes), \
            where num_dist is the vector length
                 (K in the paper) and num_nodes is the number of PC nodes in the layer.
        """

    @abstractmethod
    def backtrack(self, *args: Any, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Define routines for backtracking in EiNets, for sampling and MPE approximation.

        :param args:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def reparam_function(self) -> Tensor:
        """Return a function which transforms a tensor of unconstrained values \
            into feasible parameters."""

    @abstractmethod
    def get_shape_dict(self) -> Dict[str, int]:
        """Return param shape of the layer with description."""
