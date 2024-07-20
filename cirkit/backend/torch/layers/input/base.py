from abc import ABC
from typing import Any, Dict, Optional, Tuple

from cirkit.backend.torch.layers.base import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring
from cirkit.utils.scope import Scope


class TorchInputLayer(TorchLayer, ABC):
    """The abstract base class for input layers."""

    # NOTE: We use exactly the sae interface (F, H, B, K) -> (F, B, K) for __call__ of input layers:
    #           1. Define arity(H)=num_channels(C), reusing the H dimension.
    #           2. Define num_input_units(K)=num_vars(D), which reuses the K dimension.
    #       For dimension D (variables), we should parse the input in circuit according to the
    #       scope of the corresponding region node/symbolic input layer.

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            scope (Scope): The scope the input layer is defined on.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
            num_folds (int): The number of channels. Defaults to 1.
        """
        super().__init__(
            len(scope), num_output_units, arity=num_channels, num_folds=num_folds, semiring=semiring
        )
        self.scope = scope

    @property
    def num_variables(self) -> int:
        return self.num_input_units

    @property
    def num_channels(self) -> int:
        return self.arity

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.num_variables, self.num_channels, self.num_output_units

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "num_channels": self.num_channels,
            "num_folds": self.num_folds,
        }

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return {}
