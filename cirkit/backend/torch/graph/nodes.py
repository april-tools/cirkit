from abc import ABC, abstractmethod
from typing import Any, Tuple, TypeVar

from torch import nn


class TorchModule(nn.Module, ABC):
    def __init__(self, *, num_folds: int = 1):
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def fold_settings(self) -> Tuple[Any, ...]:
        ...


TorchModuleType = TypeVar("TorchModuleType", bound=TorchModule)
