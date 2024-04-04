from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


class Reparameterization(nn.Module, ABC):
    """The abstract base class for all reparameterizations.

    NOTE: An instance of this class can be materialized only once, and following materializations \
          are all no-op. If we do want to a true re-materialize, another instance should be \
          constructed.
    """

    def __init__(self) -> None:
        """Init class."""
        super().__init__()
        # All attributes available but empty before materialization.
        self.shape = ()
        # ANNOTATE: Specify content for empty container.
        self.dims: Tuple[int, ...] = ()  # The sum weight normalization dims; see materialize(dim=).

    shape: Tuple[int, ...]
    """The shape of the output parameter."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device of the output parameter."""

    # We forbid re-materialization by checking this flag.
    @property
    def is_materialized(self) -> bool:
        """Whether this reparameterization is already materialized."""
        # self.shape is set during materialization, so it indicates whether is materialized.
        return bool(self.shape)

    # No default value for dim, so we don't forget to provide one. We require dim because it's an
    # essential property of sum weights. If a reparam at some place is expected to be always
    # unnormalized, just explicitly pass dim=().
    # NOTE: We provide a default materialization, but subclasses are still expected to override, and
    #       therefore this is marked @abstractmethod. Yet subclasses are still expected to call
    #       super().materialize(...) to initialize shape and dims and optionally value.
    @abstractmethod
    def materialize(
        self,
        shape: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        initializer_: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> bool:
        """Materialize the internal parameter tensor(s) with given shape and initialize if required.

        Materialization (and optionally initialization) is only executed if it's not materialized \
        yet. Otherwise this function will become a silent no-op, providing safe reuse of the same \
        reparam. However, the arguments must be the same among re-materialization attempts, to \
        make sure the reuse is consistent. The return value will indicate whether there's \
        materialization happening.

        The kwarg-only dim, is used to hint the normalization of sum weights (or some input params \
        that may expect normalization). It's not always used by all layers but is required to be\
        supplied with the sum-to-1 dimension(s) so that both normalized and unnormalized reparams \
        will work under the same materialization setting.

        If an initializer_ is provided, it will be used to fill the initial value of the "output" \
        parameter, and implementations may define how the value is propagated to the internal \
        tensor(s). If no initializer is given, the internal storage will contain random memory.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied. Unnormalized implementations may choose to ignore this.
            initializer_ (Optional[Callable[[Tensor], Tensor]], optional): The function that \
                initialize a Tensor inplace while also returning the value. Leave default for no \
                initialization. Defaults to None.

        Returns:
            bool: Whether the materialization is actually performed.
        """
        assert shape, "The parameter shape cannot be empty. Use shape (1,) for scalar param."
        shape = tuple(shape)

        dims = dim if isinstance(dim, Sequence) else (dim,)
        dims = tuple(sorted(d if d >= 0 else d + len(shape) for d in dims))
        assert all(0 <= d < len(shape) for d in dims), f"dim={dim} out of range for {len(shape)}-d."

        if self.is_materialized:
            assert (
                self.shape == shape and self.dims == dims
            ), "Reparameterization cannot be re-materialized into a different configuration."
            return False

        self.shape = shape
        self.dims = dims
        # NOTE: initializer_ will not be called here because materialization is not finished yet.
        #       Subclasses should call it upon receiving the True return.
        return True

    # NOTE: Subclasses should include @torch.no_grad() to disable grad for initialization.
    @abstractmethod
    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensor(s) with the given initializer.

        This can only be called after materialization and will always overwrite whatever is \
        already in the internal param. To safely provide an initial value to a possibly reused \
        reparam, initialize through materialize() instead.

        The provided initializer_ is expected to provide an initial value for the output \
        parameter, and implementations may define how the value is transformated to initialize the \
        internal tensor(s).

        Args:
            initializer_ (Callable[[Tensor], Tensor]): The function that initialize a Tensor \
                inplace while also returning the value.
        """

    def __call__(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
