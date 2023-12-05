from typing import Callable, Generic, List, Optional, Sequence, Tuple, Union, cast
from typing_extensions import TypeVarTuple, Unpack  # TODO: in typing from 3.11

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
        # Ignore: Here we must use nn.ModuleList to register sub-modules, but we need
        #         List[Reparameterization] so that elements are properly typed.
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

    def materialize(
        self,
        shape: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        mask: Optional[Tensor] = None,
        log_mask: Optional[Tensor] = None,
    ) -> None:
        """Materialize the internal parameter tensors with given shape.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        The three kwargs, dim, mask/log_mask, are used to hint the normalization of sum weights. \
        The dim kwarg must be supplied to hint the sum-to-1 dimension, but mask/log_mask can be \
        optional and at most one can be provided.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied.
            mask (Optional[Tensor], optional): The 0/1 mask for normalization positions. None for \
                no masking. The shape must be broadcastable to shape if not None. Defaults to None.
            log_mask (Optional[Tensor], optional): The -inf/0 mask for normalization positions. \
                None for no masking. The shape must be broadcastable to shape if not None. \
                Defaults to None.
        """
        super().materialize(shape, dim=dim, mask=mask, log_mask=log_mask)
        for reparam in self.reparams:
            if not reparam.is_materialized:
                # NOTE: Passing shape to all children reparams may not be always wanted. In that
                #       case, children reparams should be materialized first, so that the following
                #       is skipped by the above if.
                reparam.materialize(shape, dim=dim, mask=mask, log_mask=log_mask)

        assert self().shape == self.shape, "The actual shape does not match the given one."

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
            # TODO: This cast is unavoidable because the type of init_value is Union[*Ts].
            init_values = (
                (init,) * len(self.reparams)
                if isinstance(init, Tensor)  # type: ignore[misc]  # Ignore: Tensor contains Any.
                else cast(Tuple[Tensor, ...], init)
            )
            for reparam, init_value in zip(self.reparams, init_values):
                # Disable: The following shuold be safe because the lambda is immediately used
                #          before the next loop iteration.  # TODO: test if what I say is correct
                # pylint: disable-next=cell-var-from-loop
                reparam.initialize(lambda x: x.copy_(init_value))

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # TODO: This cast is unavoidable because Tensor is not Ts.
        #       NOTE: params is not tuple, but it magically works with unpacking using *(...).
        params = cast(Tuple[Unpack[Ts]], (reparam() for reparam in self.reparams))
        return self.func(*params)
