from abc import ABC
from typing import TypeVar

from torch import nn


class TorchModule(nn.Module, ABC):
    def __init__(self, *, num_folds: int = 1):
        super().__init__()
        self.num_folds = num_folds


TorchModuleType = TypeVar("TorchModuleType", bound=TorchModule)
