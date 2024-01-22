from typing import Callable, Generic, List, Optional, Sequence, Tuple, Union, cast
from typing_extensions import TypeVarTuple, Unpack  # FUTURE: in typing from 3.11

import torch
from torch import Tensor, nn

from cirkit.new.reparams.leaf import LeafReparam
from cirkit.new.reparams.reparam import Reparameterization

Ts = TypeVarTuple("Ts")


# TODO: for now the solution I found is using Generic[Unpack[TypeVarTuple]], but it does not bound
#       with Tuple[Tensor, ...], and extra cast is needed. Any better solution?
class ComposedReparam(Reparameterization, Generic[Unpack[Ts]]):
    """The base class for composed reparameterization."""

    def __init__(
        self,
        *reparams: Optional[Reparameterization],
        func: Callable[[Unpack[Ts]], Tensor],
        inv_func: Optional[Callable[[Tensor], Union[Tuple[Unpack[Ts]], Tensor]]] = None,
    ) -> None:
        """Init class.

        Args:
            *reparams (Optional[Reparameterization]): The input reparameterizations to be \
                composed. If there's None, a LeafReparam will be constructed in its place, but \
                None must be provided instead of omitted so that the length is correct.
            func (Callable[[*Ts], Tensor]): The function to compose the output from the \
                parameters given by reparams.
            inv_func (Optional[Callable[[Tensor], Union[Tuple[Unpack[Ts]], Tensor]]], optional): \
                The inverse of func, used to transform the intialization. Returns one Tensor for \
                all of reparams or a tuple for each of reparams. The initializer will directly \
                pass through if no inv_func provided. Defaults to None.
        """
        super().__init__()
        # TODO: make ModuleList a generic?
        # ANNOTATE: We use List[Reparameterization] for typing so that elements are properly typed.
        # IGNORE: We must use nn.ModuleList for runtime to register sub-modules.
        self.reparams: List[Reparameterization] = nn.ModuleList(  # type: ignore[assignment]
            reparam if reparam is not None else LeafReparam() for reparam in reparams
        )
        self.func = func
        self.inv_func = inv_func

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        dtype = self.reparams[0].dtype
        assert all(
            reparam.dtype == dtype for reparam in self.reparams
        ), "The dtype of all composing reparams should be the same."
        return dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        device = self.reparams[0].device
        assert all(
            reparam.device == device for reparam in self.reparams
        ), "The device of all composing reparams should be the same."
        return device

    def materialize(self, shape: Sequence[int], /, *, dim: Union[int, Sequence[int]]) -> bool:
        """Materialize the internal parameter tensors with given shape.

        If it is already materialized, False will be returned to indicate no materialization. \
        However, a second call to materialize must give the same config, so that the underlying \
        params can indeed be reused.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        The kwarg, dim, is used to hint the normalization of sum weights. It's not always used but \
        must be supplied with the sum-to-1 dimension(s) so that it's guaranteed to be available \
        when a normalized reparam is passed as self.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied. However a subclass impl may choose to ignore this.

        Returns:
            bool: Whether the materialization is done.
        """
        if not super().materialize(shape, dim=dim):
            return False

        if len(self.reparams) > 1:
            assert all(reparam.is_materialized for reparam in self.reparams), (
                "A Reparameterization with more than one children must not be materialized before "
                "all its children are materialized."
            )
        else:
            reparam = self.reparams[0]
            if not reparam.is_materialized:
                reparam.materialize(shape, dim=dim)

        assert self().shape == self.shape, "The actual shape does not match the given one."
        return True

    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensors with the given initializer.

        Initialization will cause error if not materialized first.

        Args:
            initializer_ (Callable[[Tensor], Tensor]): A function that can initialize a tensor \
                inplace while also returning the value.
        """
        if self.inv_func is None:
            for reparam in self.reparams:
                reparam.initialize(initializer_)
        else:
            init = self.inv_func(initializer_(torch.zeros(self.shape)))
            # CAST: Expected tuple of Tensor but got Ts.
            # IGNORE: Tensor contains Any.
            init_values = (
                (init,) * len(self.reparams)
                if isinstance(init, Tensor)  # type: ignore[misc]
                else cast(Tuple[Tensor, ...], init)
            )
            for reparam, init_value in zip(self.reparams, init_values):
                # DISABLE: The following should be safe because the lambda is immediately used
                #          before the next loop iteration.  # TODO: test if what I say is correct
                # pylint: disable-next=cell-var-from-loop
                reparam.initialize(lambda x: x.copy_(init_value))

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # NOTE: params is not tuple, but generator still can be unpacked with *.
        # CAST: Expected Ts but got tuple (actually generator) of Tensor.
        params = cast(Tuple[Unpack[Ts]], (reparam() for reparam in self.reparams))
        return self.func(*params)
