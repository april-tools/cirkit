from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import Tensor, nn

# TODO: rework docstrings


# TODO: name it layer?
# TODO: what interface do we need in this very generic class?
class Layer(nn.Module, ABC):
    """Abstract layer class. Specifies functionality every layer in an EiNet should implement."""

    def __init__(self) -> None:
        """Init class."""
        super().__init__()  # TODO: do we need multi-inherit init?
        self.prob: Optional[Tensor] = None  # TODO: why None?

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset parameters to default initialization."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Get the number of params.

        Returns:
            int: the number of params
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

    # TODO: need to implement relevant things
    # TODO: should be abstract but for now NO to prevent blocking downstream
    def backtrack(self, *args: Any, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Define routines for backtracking in EiNets, for sampling and MPE approximation.

        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError
