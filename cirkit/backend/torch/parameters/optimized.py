import itertools
from typing import Any

import torch
from torch import Tensor

from cirkit.backend.torch.parameters.nodes import TorchParameterOp


class TorchEinsumParameter(TorchParameterOp):
    def __init__(
        self,
        in_shapes: tuple[tuple[int, ...], ...],
        einsum: tuple[tuple[int, ...], ...],
        num_folds: int = 1
    ):
        if len(in_shapes) != len(einsum) - 1:
            raise ValueError("Number of inputs and einsum shapes mismatch")
        idx_to_dim: dict[int, int] = {}
        for in_shape, multi_in_idx in zip(in_shapes, einsum):
            for i, einsum_idx in enumerate(multi_in_idx):
                if einsum_idx not in idx_to_dim:
                    idx_to_dim[einsum_idx] = in_shape[i]
                    continue
                if in_shape[i] != idx_to_dim[einsum_idx]:
                    raise ValueError(
                        f"Einsum shape mismatch, found {in_shape[i]} "
                        f"but expected {idx_to_dim[einsum_idx]}"
                    )
                continue
        super().__init__(*in_shapes, num_folds=num_folds)
        # Pre-compute the output shape of the einsum
        self._output_shape = tuple(
            idx_to_dim[einsum_idx] for einsum_idx in einsum[-1]
        )
        # Add fold dimension in both inputs and outputs of the einsum
        self.einsum = einsum
        self._folded_einsum = tuple(
            (0,) + tuple(map(lambda i: i + 1, einsum_idx)) for einsum_idx in einsum
        )

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["einsum"] = self.einsum
        return config

    @property
    def shape(self) -> tuple[int, ...]:
        return self._output_shape

    def forward(self, *xs: Tensor) -> Tensor:
        einsum_args = tuple(itertools.chain.from_iterable(zip(xs, self._folded_einsum[:-1])))
        return torch.einsum(*einsum_args, self.einsum[-1])
