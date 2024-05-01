# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.
from functools import cached_property
from typing import Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.params.composed import TorchBinaryOpParameter, TorchLogSoftmaxParameter

# This file is for specialized reparams designed specifically for ExpFamilyLayer.


# class TorchMeanGaussianProductParameter(TorchBinaryOpParameter):
#     def __init__(
#         self,
#         mean1: AbstractTorchParameter,
#         mean2: AbstractTorchParameter,
#         stddev1: AbstractTorchParameter,
#         stddev2: AbstractTorchParameter
#     ) -> None:
#         assert stddev1.shape[0] == stddev2.shape[0]
#         assert stddev1.shape[2] == stddev2.shape[2]
#         super().__init__(stddev1, stddev2, func=self._func)
#
#     @cached_property
#     def shape(self) -> Tuple[int, ...]:
#         return (
#             self.stddev1.shape[0],
#             self.stddev1.shape[1] * self.stddev2.shape[1],
#             self.stddev1.shape[2],
#         )
#
#     def _func(self, stddev1: Tensor, stddev2: Tensor) -> Tensor:
#         # stddev1: (D, K1, C)
#         # stddev2: (D, K2, C)
#         # out: (D, K1 * K2, C), which is the harmonic mean of the two standard deviations
#
#         # TODO (LL): this code might be numerically unstable,
#         #            but we can easily implement the log-sum-exp trick to compute the harmonic mean
#         inv_stddev1 = torch.reciprocal(stddev1)
#         inv_stddev2 = torch.reciprocal(stddev2)
#         inv_stddev = inv_stddev1.unsqueeze(dim=2) + inv_stddev2.unsqueeze(dim=1)
#         inv_stddev = inv_stddev.view(self.shape)
#         stddev = torch.reciprocal(inv_stddev)
#         return stddev
#
#
#
# class TorchStddevGaussianProductParameter(TorchBinaryOpParameter):
#     def __init__(
#         self,
#         stddev1: AbstractTorchParameter,
#         stddev2: AbstractTorchParameter
#     ) -> None:
#         assert stddev1.shape[0] == stddev2.shape[0]
#         assert stddev1.shape[2] == stddev2.shape[2]
#         super().__init__(stddev1, stddev2, func=self._func)
#
#     @cached_property
#     def shape(self) -> Tuple[int, ...]:
#         return (
#             self.stddev1.shape[0],
#             self.stddev1.shape[1] * self.stddev2.shape[1],
#             self.stddev1.shape[2],
#         )
#
#     def _func(self, stddev1: Tensor, stddev2: Tensor) -> Tensor:
#         # stddev1: (D, K1, C)
#         # stddev2: (D, K2, C)
#         # out: (D, K1 * K2, C), which is the harmonic mean of the two standard deviations
#
#         # TODO (LL): this code might be numerically unstable,
#         #            but we can easily implement the log-sum-exp trick to compute the harmonic mean
#         inv_stddev1 = torch.reciprocal(stddev1)
#         inv_stddev2 = torch.reciprocal(stddev2)
#         inv_stddev = inv_stddev1.unsqueeze(dim=2) + inv_stddev2.unsqueeze(dim=1)
#         inv_stddev = inv_stddev.view(self.shape)
#         stddev = torch.reciprocal(inv_stddev)
#         return stddev
