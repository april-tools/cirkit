# pylint: skip-file
# type: ignore
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.new.reparams.composed import ComposedReparam
from cirkit.new.reparams.normalized import LogSoftmaxReparam, SoftmaxReparam
from cirkit.new.reparams.reparam import Reparameterization
from cirkit.new.utils import flatten_dims, unflatten_dims


class BinaryReparam(ComposedReparam[Tensor, Tensor]):
    """The binary composed reparameterization."""

    def __init__(
        self,
        reparam1: Optional[Reparameterization] = None,
        reparam2: Optional[Reparameterization] = None,
        /,
        *,
        func: Callable[[Tensor, Tensor], Tensor],
        inv_func: Optional[Callable[[Tensor], Union[Tuple[Tensor, Tensor], Tensor]]] = None,
    ) -> None:
        # DISABLE: This long line is unavoidable for Args doc.
        # pylint: disable=line-too-long
        """Init class.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            func (Callable[[Tensor, Tensor], Tensor]): The function to compose the output from the \
                parameters given by reparam.
            inv_func (Optional[Callable[[Tensor], Union[Tuple[Tensor, Tensor], Tensor]]], optional): \
                The inverse of func, used to transform the intialization. The initializer will \
                directly pass through if no inv_func provided. Defaults to None.
        """
        # pylint: enable=line-too-long

        self.norm_func: Callable

        # TODO: refactor
        self.norm_func = (
            reparam1.norm_func
            if isinstance(reparam1, BinaryReparam)
            else reparam2.norm_func
            if isinstance(reparam2, BinaryReparam)
            else torch.softmax
            if (
                isinstance(reparam1, SoftmaxReparam)
                or (isinstance(reparam1, type) and issubclass(reparam1, SoftmaxReparam))
                or isinstance(reparam2, SoftmaxReparam)
                or (isinstance(reparam2, type) and issubclass(reparam2, SoftmaxReparam))
            )
            else torch.log_softmax
            if (
                isinstance(reparam1, LogSoftmaxReparam)
                or (isinstance(reparam1, type) and issubclass(reparam1, LogSoftmaxReparam))
                or isinstance(reparam2, LogSoftmaxReparam)
                or (isinstance(reparam2, type) and issubclass(reparam2, LogSoftmaxReparam))
            )
            else None
        )

        super().__init__(reparam1, reparam2, func=func, inv_func=inv_func)


class InnerLayerProductReparam(BinaryReparam):
    """reparameterization for the product of inner layers of the circuit."""

    def __init__(
        self,
        reparam1: Reparameterization,
        reparam2: Reparameterization,
        /,
    ) -> None:
        # DISABLE: This long line is unavoidable for Args doc.
        """Init class.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """

        super().__init__(reparam1, reparam2, func=self._func, inv_func=None)  # type: ignore[arg-type]

    def _func(self, param1: Tensor, param2: Tensor) -> Tensor:
        # Sum layer parameters: (K_out, 1, K_in, 1)
        # Tucker layer parameters: (K_out, 1, K_in, 1, K_in, 1)
        reshaped_param1_shape = [d for dim in zip(param1.shape, (1,) * param2.ndim) for d in dim]
        # Sum layer parameters: (1, K_out, 1, K_in)
        # Tucker layer parameters: (1, K_out, 1, K_in, 1, K_in)
        reshaped_param2_shape = [d for dim in zip((1,) * param1.ndim, param2.shape) for d in dim]

        # Sum layer parameters: (K_out1*K_out2, K_in1*K_in2)
        # Tucker layer parameters: (K_out1*K_out2, K_in1*K_in2, K_in1*K_in2)
        output_param_shape = tuple(d1 * d2 for d1, d2 in zip(param1.shape, param2.shape))

        reshaped_param1 = param1.reshape(reshaped_param1_shape)
        reshaped_param2 = param2.reshape(reshaped_param2_shape)

        kron_param = self.comp_space.mul(reshaped_param1, reshaped_param2)
        output_param = kron_param.reshape(output_param_shape)

        if self.norm_func is not None:
            output_param = unflatten_dims(
                self.norm_func(flatten_dims(output_param, dims=self.dims), dim=self.dims[0]),
                dims=self.dims,
                shape=output_param.shape,
            )
            # TODO: need nan_to_num?

        return output_param


class CategoricalProductReparam(BinaryReparam):
    """reparameterization for the product of categorical input layers."""

    def __init__(
        self,
        reparam1: Reparameterization,
        reparam2: Reparameterization,
        /,
    ) -> None:
        # DISABLE: This long line is unavoidable for Args doc.
        """Init class.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """

        super().__init__(reparam1, reparam2, func=self._func, inv_func=None)  # type: ignore[arg-type]

    def _func(self, param1: Tensor, param2: Tensor) -> Tensor:
        # (arity, K_out_1, 1, K_in, cat)
        reshaped_param1 = param1.unsqueeze(2)
        # (arity, 1, K_out_2, K_in, cat)
        reshaped_param2 = param2.unsqueeze(1)
        # (arity, K_out_1*K_out_2, K_in, cat)
        output_param = self.comp_space.mul(reshaped_param1, reshaped_param2).view(
            param1.shape[0], -1, param1.shape[2], param1.shape[3]
        )

        if self.norm_func is not None:
            assert len(self.dims) == 1
            output_param = self.norm_func(output_param, dim=self.dims[0])
            # TODO: need nan_to_num?
        return output_param


class EFNormalProductReparam(BinaryReparam):
    """reparameterization for the product of exponential family normal input layers."""

    def __init__(
        self,
        reparam1: Reparameterization,
        reparam2: Reparameterization,
        /,
    ) -> None:
        # DISABLE: This long line is unavoidable for Args doc.
        """Init class.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """

        super().__init__(reparam1, reparam2, func=self._func, inv_func=None)  # type: ignore[arg-type]

    def _func(self, param1: Tensor, param2: Tensor) -> Tensor:
        # (arity, K_out_1, 1, 2, K_in)
        reshaped_param1 = param1.unsqueeze(2)
        # (arity, 1, K_out_2, 2, K_in)
        reshaped_param2 = param2.unsqueeze(1)
        # (arity, K_out_1*K_out_2, 2, K_in)
        output_param = self.comp_space.sum(
            torch.add, reshaped_param1, reshaped_param2, dim=(1, 2), keepdim=True
        ).view(param1.shape[0], -1, param1.shape[2], param1.shape[3])

        return output_param
