# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

import torch
from torch import Tensor

from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.params.composed import TorchBinaryOpParameter, TorchLogSoftmaxParameter

# This file is for specialized reparams designed specifically for ExpFamilyLayer.

TorchCategoricalParameter = TorchLogSoftmaxParameter


class TorchGaussianMeanProductParameter(TorchBinaryOpParameter):
    """Reparameterization for product of Exponential Family.

    This is designed to do the "kronecker concat" with flattened suff stats:
        - Expected input: (H, K_1, *S_1), (H, K_2, *S_2);
        - Will output: (H, K_1*K_2, flatten(S_1)+flatten(S_2)).
    """

    def __init__(
        self,
        param1: AbstractTorchParameter,
        param2: AbstractTorchParameter,
        /,
    ) -> None:
        """Init class.

        Args:
            param1 (AbstractTorchParameter): The first input reparam to be composed.
            param2 (AbstractTorchParameter): The second input reparam to be composed.
        """
        super().__init__(param1, param2, func=self._func)

    @classmethod
    def _func(cls, param1: Tensor, param2: Tensor) -> Tensor:
        # shape (H, K, *S) -> (H, K, S) -> (H, K, 1, S).
        param1 = param1.flatten(start_dim=2).unsqueeze(dim=2)
        # shape (H, K, *S) -> (H, K, S) -> (H, 1, K, S).
        param2 = param2.flatten(start_dim=2).unsqueeze(dim=1)
        # IGNORE: broadcast_tensors is not typed.
        # shape (H, K, K, S+S) -> (H, K*K, S+S).
        return torch.cat(
            torch.broadcast_tensors(param1, param2), dim=-1  # type: ignore[no-untyped-call,misc]
        ).flatten(start_dim=1, end_dim=2)
