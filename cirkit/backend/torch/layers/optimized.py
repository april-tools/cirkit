from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from cirkit.backend.torch.layers import TorchSumLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring


class TorchTensorDotLayer(TorchSumLayer):
    """The sum layer for dense sum within a layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        num_folds: int = 1,
        weight: TorchParameter,
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_outpfrom functools import cached_propertyut_units (int): The number of output units.
            num_folds (int): The number of channels. Defaults to 1.
            weight (TorchParameter): The reparameterization for layer parameters.
        """
        num_contract_units = weight.shape[1]
        num_batch_units = num_input_units // num_contract_units
        assert weight.num_folds == num_folds
        assert num_input_units % weight.shape[1] == 0
        assert num_output_units == weight.shape[0] * num_batch_units
        super().__init__(
            num_input_units, num_output_units, arity=1, num_folds=num_folds, semiring=semiring
        )
        self._num_contract_units = num_contract_units
        self._num_batch_units = num_batch_units
        self.weight = weight

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "num_folds": self.num_folds,
        }

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, self._num_batch_units

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        # x: (F, H=1, B, Ki) -> (F, B, Ki)
        x = x.squeeze(dim=1)
        # x: (F, B, Ki) -> (F, B, Kj, Kq) -> (F, B, Kq, Kj)
        x = x.view(x.shape[0], x.shape[1], self._num_contract_units, self._num_batch_units)
        x = x.permute(0, 1, 3, 2)
        # weight: (F, Kk, Kj)
        weight = self.weight()
        # y: (F, B, Kq, Kj)
        y = self.semiring.einsum(
            "fbqj,fkj->fbqk", inputs=(x,), operands=(weight,), dim=-1, keepdim=True
        )
        # return y: (F, B, Kq * Kj) = (F, B, Ko)
        return y.view(y.shape[0], y.shape[1], self.num_output_units)
