from abc import ABC
from typing import Any, Dict

from cirkit.backend.torch.layers.inner import TorchInnerLayer


class TorchSumProductLayer(TorchInnerLayer, ABC):
    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
            "num_folds": self.num_folds,
        }
