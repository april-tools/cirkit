from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.params.composed import TorchOpParameter


class TorchFoldIdxParameter(TorchOpParameter):
    def __init__(self, opd: AbstractTorchParameter, fold_idx: int) -> None:
        assert 0 <= fold_idx < opd.num_folds
        super().__init__(num_folds=1)
        self.opd = opd
        if isinstance(fold_idx, int):
            fold_idx = [fold_idx]
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


class TorchFoldParameter(TorchOpParameter):
    def __init__(
        self, *opds: AbstractTorchParameter, fold_idx: Optional[Union[int, List[int]]] = None
    ) -> None:
        assert len(set(opd.shape for opd in opds)) == 1
        if fold_idx is not None:
            if isinstance(fold_idx, int):
                fold_idx = [fold_idx]
            fold_idx = torch.tensor(fold_idx)
            num_folds = len(fold_idx)
        else:
            num_folds = sum(opd.num_folds for opd in opds)
        super().__init__(num_folds=num_folds)
        self.opds = opds
        self.fold_idx = fold_idx

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
        # return: (F, d1, ..., dn), with possibly F = F'
        if self.fold_idx is not None:
            return x[self.fold_idx]
        return x

    def forward(self) -> Tensor:
        x = torch.cat([opd() for opd in self.opds], dim=0)
        return self._forward_impl(x)
