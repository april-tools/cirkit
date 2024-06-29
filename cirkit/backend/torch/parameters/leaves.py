from typing import Any, Callable, Dict, List, Optional, Tuple, Union, final

import torch
from torch import Tensor, nn

from cirkit.backend.torch.parameters.parameter import TorchParameterLeaf


class TorchTensorParameter(TorchParameterLeaf):
    """The leaf in reparameterizations that holds the parameter Tensor."""

    def __init__(
        self,
        *shape: int,
        num_folds: int = 1,
        requires_grad: bool = True,
        initializer_: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """Init class."""
        super().__init__(num_folds=num_folds)
        self._shape = shape
        self._ptensor: Optional[nn.Parameter] = None
        self._requires_grad = requires_grad
        self.initializer_ = nn.init.normal_ if initializer_ is None else initializer_

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        assert self._ptensor is not None
        return self._ptensor.dtype

    @property
    def device(self) -> torch.device:
        assert self._ptensor is not None
        return self._ptensor.device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
        if self._ptensor is not None:
            self._ptensor.requires_grad = value

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.shape, self.requires_grad

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return dict(shape=self._shape, num_folds=self.num_folds, requires_grad=self._requires_grad)

    @property
    def is_initialized(self) -> bool:
        return self._ptensor is not None

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize the internal parameter tensor with the given initializer."""
        if self._ptensor is None:
            shape = (self.num_folds, *self._shape)
            self._ptensor = nn.Parameter(torch.empty(*shape), requires_grad=self._requires_grad)
            self.initializer_(self._ptensor.data)
            return
        self.initializer_(self._ptensor.data)

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        assert self._ptensor is not None
        return self._ptensor


@final
class TorchPointerParameter(TorchParameterLeaf):
    def __init__(
        self, parameter: TorchTensorParameter, *, fold_idx: Optional[Union[int, List[int]]] = None
    ) -> None:
        if fold_idx is None:
            num_folds = parameter.num_folds
        elif isinstance(fold_idx, int):
            assert 0 <= fold_idx < parameter.num_folds
            if fold_idx == 0 and parameter.num_folds == 1:
                fold_idx = None
                num_folds = parameter.num_folds
            else:
                fold_idx = [fold_idx]
                num_folds = 1
        else:
            assert isinstance(fold_idx, list)
            assert all(0 <= i < parameter.num_folds for i in fold_idx)
            if fold_idx == list(range(parameter.num_folds)):
                fold_idx = None
                num_folds = parameter.num_folds
            else:
                num_folds = len(fold_idx)
        assert not isinstance(parameter, TorchPointerParameter)
        super().__init__(num_folds=num_folds)
        self._parameter = parameter
        self._fold_idx = fold_idx

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the output parameter."""
        return self._parameter.shape

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.shape, id(self._parameter)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(fold_idx=self._fold_idx)

    def deref(self) -> TorchTensorParameter:
        return self._parameter

    @property
    def fold_idx(self) -> Optional[int]:
        return self._fold_idx

    def forward(self) -> Tensor:
        x = self._parameter()
        if self._fold_idx is None:
            return x
        return x[self.fold_idx]
