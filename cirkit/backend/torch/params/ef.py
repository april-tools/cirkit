from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.params.base import AbstractTorchParameter


class TorchMultiParameters(nn.Module, ABC):
    def __init__(self, **params: List[AbstractTorchParameter]):
        super().__init__()
        self.params = params

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        ps = [p for rs in self.params.values() for p in rs]
        dtype = ps[0].dtype
        assert all(
            p.dtype == dtype for p in ps
        ), "The dtype of all composing parameters should be the same."
        return dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        ps = [p for rs in self.params.values() for p in rs]
        device = ps[0].device
        assert all(
            p.device == device for p in ps
        ), "The device of all composing parameters should be the same."
        return device

    @abstractmethod
    def forward(self) -> Tensor:
        ...


class TorchMeanGaussianProductParameter(TorchMultiParameters):
    def __init__(
        self, mean_ls: List[AbstractTorchParameter], stddev_ls: List[AbstractTorchParameter]
    ) -> None:
        assert len(mean_ls) > 1 and len(mean_ls) == len(stddev_ls)
        assert len(set((m.shape[0], m.shape[2]) for m in mean_ls)) == 1
        assert len(set((s.shape[0], s.shape[2]) for s in stddev_ls)) == 1
        assert all(
            m.shape[0] == s.shape[0] and m.shape[2] == s.shape[2]
            for m, s in zip(mean_ls, stddev_ls)
        )
        super().__init__(mean_ls=mean_ls, stddev_ls=stddev_ls)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        mean_ls = self.params["mean_ls"]
        cross_dim = np.prod([m.shape[1] for m in mean_ls])
        return mean_ls[0].shape[0], cross_dim, mean_ls[0].shape[2]

    def forward(self) -> Tensor:
        # *mean_ls: (D, Ki, C)
        # *stddev_ls: (D, Ki, C)
        # return: (D, K1 * ... * Kn, C)
        # TODO (LL): this code might be numerically unstable and/or infefficient,
        # - It might be possible to find another more efficient implementation
        # - We can easily implement the same operations over the log-sum-exp:sum semiring
        mean_ls: List[AbstractTorchParameter] = self.params["mean_ls"]
        stddev_ls: List[AbstractTorchParameter] = self.params["stddev_ls"]
        num_products = len(mean_ls)
        ms = [m() for m in mean_ls]
        ss = [s() for s in stddev_ls]
        vs = [torch.square(s) for s in ss]
        div_m = None
        sum_rec_v = None
        for i in range(num_products):
            if i == 0:
                sum_rec_v = torch.reciprocal(vs[0])
                div_m = ms[0] * sum_rec_v
                continue
            oth_rec_v = torch.reciprocal(vs[i])
            cross_dim = sum_rec_v.shape[1] * oth_rec_v.shape[1]
            oth_div_m = ms[i] * oth_rec_v
            sum_rec_v = sum_rec_v.unsqueeze(dim=2) + oth_rec_v.unsqueeze(dim=1)
            div_m = div_m.unsqueeze(dim=2) + oth_div_m.unsqueeze(dim=1)
            sum_rec_v = sum_rec_v.view(sum_rec_v.shape[0], cross_dim, sum_rec_v.shape[3])
            div_m = div_m.view(div_m.shape[0], cross_dim, div_m.shape[3])
        mean = div_m * torch.reciprocal(sum_rec_v)
        return mean


class TorchStddevGaussianProductParameter(TorchMultiParameters):
    def __init__(self, stddev_ls: List[AbstractTorchParameter]) -> None:
        assert len(stddev_ls) > 1
        assert len(set((s.shape[0], s.shape[2]) for s in stddev_ls)) == 1
        super().__init__(stddev_ls=stddev_ls)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        stddev_ls = self.params["stddev_ls"]
        cross_dim = np.prod([m.shape[1] for m in stddev_ls])
        return stddev_ls[0].shape[0], cross_dim, stddev_ls[0].shape[2]

    def forward(self) -> Tensor:
        # *mean_ls: (D, Ki, C)
        # *stddev_ls: (D, Ki, C)
        # return: (D, K1 * ... * Kn, C)
        # TODO (LL): this code might be numerically unstable and/or infefficient,
        # - It might be possible to find another more efficient implementation
        # - We can easily implement the same operations over the log-sum-exp:sum semiring
        stddev_ls: List[AbstractTorchParameter] = self.params["stddev_ls"]
        num_products = len(stddev_ls)
        ss = [s() for s in stddev_ls]
        vs = [torch.square(s) for s in ss]
        sum_rec_v = None
        for i in range(num_products):
            if i == 0:
                sum_rec_v = torch.reciprocal(vs[0])
                continue
            oth_rec_v = torch.reciprocal(vs[i])
            cross_dim = sum_rec_v.shape[1] * oth_rec_v.shape[1]
            sum_rec_v = sum_rec_v.unsqueeze(dim=2) + oth_rec_v.unsqueeze(dim=1)
            sum_rec_v = sum_rec_v.view(sum_rec_v.shape[0], cross_dim, sum_rec_v.shape[3])
        variance = torch.reciprocal(sum_rec_v)
        return torch.sqrt(variance)


class TorchLogPartitionGaussianProductParameter(TorchMultiParameters):
    def __init__(
        self, mean_ls: List[AbstractTorchParameter], stddev_ls: List[AbstractTorchParameter]
    ) -> None:
        assert len(mean_ls) > 1 and len(mean_ls) == len(stddev_ls)
        assert len(set((m.shape[0], m.shape[2]) for m in mean_ls)) == 1
        assert len(set((s.shape[0], s.shape[2]) for s in stddev_ls)) == 1
        assert all(
            m.shape[0] == s.shape[0] and m.shape[2] == s.shape[2]
            for m, s in zip(mean_ls, stddev_ls)
        )
        super().__init__(mean_ls=mean_ls, stddev_ls=stddev_ls)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        mean_ls = self.params["mean_ls"]
        cross_dim = np.prod([m.shape[1] for m in mean_ls])
        return (cross_dim,)

    def forward(self) -> Tensor:
        # *mean_ls: (D, Ki, C)
        # *stddev_ls: (D, Ki, C)
        # return: (D, K1 * ... * Kn, C)
        # TODO (LL): this code might be numerically unstable and/or infefficient,
        # - It might be possible to find another more efficient implementation
        # - We can easily implement the same operations over the log-sum-exp:sum semiring
        mean_ls: List[AbstractTorchParameter] = self.params["mean_ls"]
        stddev_ls: List[AbstractTorchParameter] = self.params["stddev_ls"]
        num_products = len(mean_ls)
        ms = [m() for m in mean_ls]
        ss = [s() for s in stddev_ls]
        vs = [torch.square(s) for s in ss]
        div_m = None
        div_q = None
        sum_rec_v = None
        log_prod_v = None
        for i in range(num_products):
            if i == 0:
                sum_rec_v = torch.reciprocal(vs[0])
                div_m = ms[0] * sum_rec_v
                div_q = div_m * ms[0]
                log_prod_v = torch.log(vs[0])
                continue
            oth_rec_v = torch.reciprocal(vs[i])
            cross_dim = sum_rec_v.shape[1] * oth_rec_v.shape[1]
            oth_div_m = ms[i] * oth_rec_v
            oth_div_q = oth_div_m * ms[i]
            oth_log_prod_v = torch.log(vs[i])
            sum_rec_v = sum_rec_v.unsqueeze(dim=2) + oth_rec_v.unsqueeze(dim=1)
            div_m = div_m.unsqueeze(dim=2) + oth_div_m.unsqueeze(dim=1)
            div_q = div_q.unsqueeze(dim=2) + oth_div_q.unsqueeze(dim=1)
            log_prod_v = log_prod_v.unsqueeze(dim=2) + oth_log_prod_v.unsqueeze(dim=1)
            sum_rec_v = sum_rec_v.view(sum_rec_v.shape[0], cross_dim, sum_rec_v.shape[3])
            div_m = div_m.view(div_m.shape[0], cross_dim, div_m.shape[3])
            div_q = div_q.view(div_q.shape[0], cross_dim, div_q.shape[3])
            log_prod_v = log_prod_v.view(log_prod_v.shape[0], cross_dim, log_prod_v.shape[3])
        variance = torch.reciprocal(sum_rec_v)
        mean = div_m * variance
        p0 = (num_products - 1) * np.log(2.0 * np.pi)
        p1 = torch.log(variance) - log_prod_v
        p2 = torch.log(div_q - torch.square(mean) * sum_rec_v)
        log_partition = torch.sum(-0.5 * (p0 - p1 + p2), dim=[0, 2])
        return log_partition


# class TorchMultiParameters(nn.Module, ABC):
#     def __init__(self, **params: List[AbstractTorchParameter]):
#         super().__init__()
#         self.params = params
#
#     @abstractmethod
#     def shape(self, name: str) -> Tuple[int, ...]:
#         ...
#
#     @property
#     def dtype(self) -> torch.dtype:
#         """The dtype of the output parameter."""
#         ps = [p for rs in self.params.values() for p in rs]
#         dtype = ps[0].dtype
#         assert all(
#             p.dtype == dtype for p in ps
#         ), "The dtype of all composing parameters should be the same."
#         return dtype
#
#     @property
#     def device(self) -> torch.device:
#         """The device of the output parameter."""
#         ps = [p for rs in self.params.values() for p in rs]
#         device = ps[0].device
#         assert all(
#             p.device == device for p in ps
#         ), "The device of all composing parameters should be the same."
#         return device
#
#     @abstractmethod
#     def forward(self) -> Dict[str, Tensor]:
#         ...


# class TorchSelectorParameter(AbstractTorchParameter):
#     def __init__(self, multi_param: TorchMultiParameters, name: str):
#         super().__init__()
#         self.multi_param = multi_param
#         self.name = name
#
#     @cached_property
#     def shape(self) -> Tuple[int, ...]:
#         return self.multi_param.shape(self.name)
#
#     @property
#     def dtype(self) -> torch.dtype:
#         return self.multi_param.dtype
#
#     @property
#     def device(self) -> torch.device:
#         return self.multi_param.device
#
#     def forward(self) -> Tensor:
#         return self.multi_param()[self.name]


# class TorchGaussianProductParameters(TorchMultiParameters):
#     def __init__(
#         self, mean_ls: List[AbstractTorchParameter], stddev_ls: List[AbstractTorchParameter]
#     ) -> None:
#         assert len(mean_ls) > 1 and len(mean_ls) == len(stddev_ls)
#         assert len(set((m.shape[0], m.shape[2]) for m in mean_ls)) == 1
#         assert len(set((s.shape[0], s.shape[2]) for s in stddev_ls)) == 1
#         assert all(
#             m.shape[0] == s.shape[0] and m.shape[2] == s.shape[2]
#             for m, s in zip(mean_ls, stddev_ls)
#         )
#         super().__init__(mean_ls=mean_ls, stddev_ls=stddev_ls)
#
#     def shape(self, name: str) -> Tuple[int, ...]:
#         mean_ls = self.params["mean_ls"]
#         cross_dim = np.prod([m.shape[1] for m in mean_ls])
#         if name == "log_partition":
#             return (cross_dim,)
#         return mean_ls[0].shape[0], cross_dim, mean_ls[0].shape[2]
#
#     def forward(self) -> Dict[str, Tensor]:
#         # *mean_ls: (D, Ki, C)
#         # *stddev_ls: (D, Ki, C)
#         # return: (D, K1 * ... * Kn, C)
#         # TODO (LL): this code might be numerically unstable and/or infefficient,
#         # - It might be possible to find another more efficient implementation
#         # - We can easily implement the same operations over the log-sum-exp:sum semiring
#         mean_ls: List[AbstractTorchParameter] = self.params["mean_ls"]
#         stddev_ls: List[AbstractTorchParameter] = self.params["stddev_ls"]
#         num_products = len(mean_ls)
#         ms = [m() for m in mean_ls]
#         ss = [s() for s in stddev_ls]
#         vs = [torch.square(s) for s in ss]
#         div_m = None
#         div_q = None
#         sum_rec_v = None
#         log_prod_v = None
#         for i in range(num_products):
#             if i == 0:
#                 sum_rec_v = torch.reciprocal(vs[0])
#                 div_m = ms[0] * sum_rec_v
#                 div_q = ms[0] * div_m
#                 log_prod_v = torch.log(vs[0])
#                 continue
#             oth_rec_v = torch.reciprocal(vs[i])
#             cross_dim = sum_rec_v.shape[1] * oth_rec_v.shape[1]
#             oth_div_m = ms[i] * oth_rec_v
#             oth_div_q = ms[i] * oth_div_m
#             oth_log_prod_v = torch.log(vs[i])
#             sum_rec_v = sum_rec_v.unsqueeze(dim=2) + oth_rec_v.unsqueeze(dim=1)
#             div_m = div_m.unsqueeze(dim=2) + oth_div_m.unsqueeze(dim=1)
#             div_q = div_q.unsqueeze(dim=2) + oth_div_q.unsqueeze(dim=1)
#             log_prod_v = log_prod_v.unsqueeze(dim=2) + oth_log_prod_v.unsqueeze(dim=1)
#             sum_rec_v = sum_rec_v.view(sum_rec_v.shape[0], cross_dim, sum_rec_v.shape[3])
#             div_m = div_m.view(div_m.shape[0], cross_dim, div_m.shape[3])
#             div_q = div_q.view(div_q.shape[0], cross_dim, div_q.shape[3])
#             log_prod_v = log_prod_v.view(log_prod_v.shape[0], cross_dim, log_prod_v.shape[3])
#         variance = torch.reciprocal(sum_rec_v)
#         mean = div_m * variance
#         p0 = (num_products - 1) * np.log(2.0 * np.pi)
#         p1 = torch.log(variance) - log_prod_v
#         p2 = torch.log(div_q - torch.square(mean) * sum_rec_v)
#         log_partition = torch.sum(-0.5 * (p0 + p1 + p2), dim=[0, 2])
#         return dict(mean=mean, stddev=torch.sqrt(variance), log_partition=log_partition)
