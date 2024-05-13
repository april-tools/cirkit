from typing import Dict, List, Optional, Tuple, Union, final

import torch
from torch import Tensor

from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.params.composed import TorchOpParameter


@final
class TorchFoldIdxParameter(TorchOpParameter):
    def __init__(self, opd: AbstractTorchParameter, fold_idx: int) -> None:
        assert 0 <= fold_idx < opd.num_folds
        super().__init__(num_folds=1)
        self.opd = opd
        self.fold_idx = fold_idx

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the output parameter."""
        return self.opd.shape

    @property
    def params(self) -> Dict[str, "AbstractTorchParameter"]:
        """The other parameters this parameter depends on."""
        return dict(opd=self.opd)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        return self.opd.dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        return self.opd.device

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x: (F', d1, ..., dn)
        # return: (1, d1, ..., dn)
        return x[self.fold_idx].unsqueeze(dim=0)

    def forward(self) -> Tensor:
        return self._forward_impl(self.opd())


@final
class TorchFoldParameter(TorchOpParameter):
    def __init__(self, *opds: AbstractTorchParameter, fold_idx: List[int]) -> None:
        assert len(set(opd.shape for opd in opds)) == 1
        num_folds = sum(opd.num_folds for opd in opds)
        assert len(fold_idx) == num_folds
        super().__init__(num_folds=num_folds)
        self.opds = opds
        fold_idx = None if fold_idx == list(range(num_folds)) else torch.tensor(fold_idx)
        self.register_buffer("_fold_idx", fold_idx)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the output parameter."""
        return self.opds[0].shape

    @property
    def params(self) -> Dict[str, "AbstractTorchParameter"]:
        """The other parameters this parameter depends on."""
        return {f"opd:{i}": opd for i, opd in enumerate(self.opds)}

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        dtype = self.opds[0].dtype
        assert all(
            opd.dtype == dtype for opd in self.opds
        ), "The dtype of all composing parameters should be the same."
        return dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        device = self.opds[0].device
        assert all(
            opd.device == device for opd in self.opds
        ), "The device of all composing parameters should be the same."
        return device

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x: (F', d1, ..., dn)
        # return: (F, d1, ..., dn)
        if self._fold_idx is None:
            return x
        return x[self._fold_idx]

    def forward(self) -> Tensor:
        if len(self.opds) == 1:
            (opd,) = self.opds
            x = opd()
        else:
            x = torch.cat([opd() for opd in self.opds], dim=0)
        return self._forward_impl(x)
