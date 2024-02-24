from typing import Callable, Generic, List, Optional, Sequence, Tuple, Union, cast
from typing_extensions import TypeVarTuple, Unpack  # FUTURE: in typing from 3.11

import torch
from torch import Tensor, nn

from cirkit.new.reparams.reparam import Reparameterization

Ts = TypeVarTuple("Ts")


# TODO: for now the solution I found is using Generic[Unpack[TypeVarTuple]], but it does not bound
#       with Tuple[Tensor, ...], and extra cast is needed. Any better solution?
class ComposedReparam(Reparameterization, Generic[Unpack[Ts]]):
    """The base class for composed reparameterization."""

    def __init__(
        self,
        *reparams: Reparameterization,
        func: Callable[[Unpack[Ts]], Tensor],
        inv_func: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> None:
        """Init class.

        Args:
            *reparams (Reparameterization): The input reparam(s) to be composed.
            func (Callable[[*Ts], Tensor]): The function to compose the output from the parameters \
                given by the input reparam(s).
            inv_func (Optional[Callable[[Tensor], Tensor]], optional): The inverse of func (in \
                unary form), used to transform the intialization. For n-ary func, this inverse may \
                not be well-defined and is therefore NOT used. Defaults to lambda x: x.
        """
        super().__init__()
        # TODO: make ModuleList a generic?
        # ANNOTATE: We use List[Reparameterization] for typing so that elements are properly typed.
        # IGNORE: We must use nn.ModuleList for runtime to register sub-modules.
        self.reparams: List[Reparameterization] = nn.ModuleList(  # type: ignore[assignment]
            reparams
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
        initializer_: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> bool:
        """Materialize the internal parameter tensor(s) with given shape and initialize if required.

        If self contains more than one input reparam, the materialization will not propagate to \
        the inputs, which means the inputs are expected to be materialized before self. This is \
        because we don't know how to decompose the given shape and dim. And thus, initialization \
        will also be skipped in this case (see also description for initialize()).

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
        if not super().materialize(shape, dim=dim):  # super() does not use initializer_.
            return False

        if len(self.reparams) > 1:
            # Only do checks but no actual work.
            assert all(reparam.is_materialized for reparam in self.reparams), (
                "A ComposedReparam with more than one input must not be materialized before all "
                "its inputs are materialized."
            )
        else:
            reparam = self.reparams[0]
            if not reparam.is_materialized:
                # NOTE: We assume func does not change shape. Otherwise the input reparam must be
                #       materialized before self.
                reparam.materialize(shape, dim=dim)
                # initializer_ cannot be directly passed through, must go through transformation.
                if initializer_ is not None:
                    self.initialize(initializer_)

        assert self().shape == self.shape, "The actual shape does not match the given one."
        return True

    @torch.no_grad()
    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensor(s) with the given initializer.

        This can only be called after materialization and will always overwrite whatever is \
        already in the internal param. To safely provide an initial value to a possibly reused \
        reparam, initialize through materialize() instead.

        The provided initializer_ is expected to provide an initial value for the output \
        parameter, and the value will be transformed through inv_func (as defined in __init__) and \
        passed to the input reparam, if there's only one input.
        In case of multiple inputs, the inputs are expected to initialized beforehand, as also in \
        materialization, and this function will silently become a no-op. This is because it's \
        generally difficult to define the inverse of n-ary functions and we just skip it for safety.

        Args:
            initializer_ (Callable[[Tensor], Tensor]): The function that initialize a Tensor \
                inplace while also returning the value.
        """
        if len(self.reparams) > 1:
            return

        # Construct a tmp Tensor -> initialize it -> transform by inv_func -> fill into x.
        self.reparams[0].initialize(
            lambda x: x.copy_(self.inv_func(initializer_(x.new_empty(self.shape))))
        )

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # NOTE: params is not tuple, but generator still can be unpacked with *.
        # CAST: Expected Ts but got tuple (actually generator) of Tensor.
        params = cast(Tuple[Unpack[Ts]], (reparam() for reparam in self.reparams))
        return self.func(*params)
