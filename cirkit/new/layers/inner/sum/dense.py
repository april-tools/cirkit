# mypy: disable-error-code="misc"
from typing import Literal

import torch
from torch import Tensor

from cirkit.new.layers.inner.sum.sum import SumLayer
from cirkit.new.reparams import Reparameterization


class DenseLayer(SumLayer):
    """The sum layer for dense sum within a layer."""

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be input**arity.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        assert arity == 1, "DenseLayer must have arity=1. For arity>1, use MixingLayer."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.params = reparam
        if self.params.materialize((num_output_units, num_input_units), dim=1):
            self.reset_parameters()  # Only reset if newly materialized.

    def _forward_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("oi,...i->...o", self.params(), x)  # shape (*B, Ki) -> (*B, Ko).

    # pylint: disable=no-self-use
    def _product_forward_1(self, param: Tensor, x: Tensor) -> Tensor:
        return torch.einsum("oi,...i->...o", param, x)  # shape (*B, Ki) -> (*B, Ko).

    # pylint: disable=no-self-use
    def _product_forward_2(self, param: Tensor, x: Tensor) -> Tensor:
        return torch.einsum("bi...,oi->b...o", x, param)  # shape (*B, Ki, ...) -> (*B, ..., Ko).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        x = x.squeeze(dim=0)  # shape (H=1, *B, K) -> (*B, K).

        # when the parameter is that of the product of two circuits
        # shape of x is (*B, K_in_1, K_in_2, ...)
        if isinstance(self.params(), list):
            batch = x.shape[0]
            # params [(K_out_i, K_in_i),(K_out_j, K_in_j)...]
            k_out = [param.shape[0] for param in self.params()]  # type: ignore[misc]
            num_params = len(self.params())
            assert (
                len(x.shape) == num_params + 1
            ), "input shape does not match the number of parameters"

            # kron(param1, param2) @ x = param_1 @ x @ param_2.T
            x_mid = self.comp_space.sum(
                lambda x: self._product_forward_1(self.params()[0], x), x, dim=-1, keepdim=True
            )
            x_nxt = self.comp_space.sum(
                lambda x: self._product_forward_2(self.params()[1], x), x_mid, dim=-2, keepdim=True
            )

            # x_nxt @ param_i.T
            if num_params > 2:
                for i in range(2, num_params):
                    param_i = self.params()[i]
                    x_nxt = self.comp_space.sum(
                        lambda x, param=param_i: self._product_forward_2(param, x),
                        x_nxt,
                        dim=-(i + 1),
                        keepdim=True,
                    )

            return x_nxt.reshape(batch, *k_out)  # (B, K_out_1, K_out_2, ...)

        # when the parameter is unary
        assert isinstance(self.params(), Tensor), "The parameter is not unary"
        return self.comp_space.sum(self._forward_linear, x, dim=-1, keepdim=True)  # shape (*B, K).
