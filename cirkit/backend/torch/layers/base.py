from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cached_property
from typing import Any

import torch
from torch import Tensor

from cirkit.backend.torch.graph.modules import AbstractTorchModule
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring, SumProductSemiring


class TorchLayer(AbstractTorchModule, ABC):
    """The abstract base class for all layers implemented in torch."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ) -> None:
        """Initialize a layer.

        Args:
            num_input_units: The number of input units.
            num_output_units: The number of output units.
            arity: The arity of the layer. Defaults to 1.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].
            num_folds: The number of folds. Defaults to 1.

        Raises:
            ValueError: If the number of input units is negative.
            ValueError: If the number of output units is not positive.
            VAlueError: If the arity is not positive.
        """
        if num_input_units < 0:
            raise ValueError("The number of input units must be non-negative")
        if num_output_units <= 0:
            raise ValueError("The number of output units must be positive")
        if arity <= 0:
            raise ValueError("The arity must be positive")
        super().__init__(num_folds=num_folds)
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity
        self.semiring = semiring if semiring is not None else SumProductSemiring

    @property
    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        """Retrieves the configuration of the layer, i.e., a dictionary mapping hyperparameters
        of the layer to their values. The hyperparameter names must match the argument names in
        the ```__init__``` method.

        Returns:
            Mapping[str, Any]: A dictionary from hyperparameter names to their value.
        """

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        """Retrieve the torch parameters of the layer, i.e., a dictionary mapping the names of
        the torch parameters to the actual torch parameter instance. The parameter names must
        match the argument names in the```__init__``` method.

        Returns:
            Mapping[str, TorchParameter]: A dictionary from parameter names to the corresponding
                torch parameter instance.
        """
        return {}

    @property
    def sub_modules(self) -> Mapping[str, "TorchLayer"]:
        """Retrieve a dictionary mapping string identifiers to torch sub-module layers.,
        that must be passed to the ```__init__``` method of the top-level layer

        Returns:
            A dictionary of torch modules.
        """
        return {}

    @cached_property
    def num_parameters(self) -> int:
        """Retrieve the number of scalar parameters. Note that if a parameter is complex-valued,
        this will double count them.

        Returns:
            The number of scalar parameters.
        """
        return sum(2 * p.numel() if torch.is_complex(p) else p.numel() for p in self.parameters())

    @cached_property
    def num_buffers(self) -> int:
        """Retrieve the number of scalar buffers. Note that if a buffer is complex-valued,
        this will double count them.

        Returns:
            The number of scalar buffers.
        """
        return sum(2 * b.numel() if torch.is_complex(b) else b.numel() for b in self.buffers())

    def __call__(self, x: Tensor) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    def extra_repr(self) -> str:
        return (
            "  ".join(
                [
                    f"folds: {self.num_folds}",
                    f"arity: {self.arity}",
                    f"input-units: {self.num_input_units}",
                    f"output-units: {self.num_output_units}",
                ]
            )
            + "\n"
            + f"input-shape: {(self.num_folds, self.arity, -1, self.num_input_units)}"
            + "\n"
            + f"output-shape: {(self.num_folds, -1, self.num_output_units)}"
        )
