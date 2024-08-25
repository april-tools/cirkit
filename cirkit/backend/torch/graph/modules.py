import abc
import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union, cast

import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph.address_book import AddressBook, FoldIndexInfo
from cirkit.utils.algorithms import DiAcyclicGraph, NodeType


class AbstractTorchModule(nn.Module, ABC):
    def __init__(self, *, num_folds: int = 1):
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def fold_settings(self) -> Tuple[Any, ...]:
        ...


TorchModule = TypeVar("TorchModule", bound=AbstractTorchModule)


class TorchDiAcyclicGraph(nn.Module, DiAcyclicGraph[TorchModule], ABC):
    def __init__(
        self,
        modules: List[TorchModule],
        in_modules: Dict[TorchModule, List[TorchModule]],
        outputs: List[TorchModule],
        *,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(nn.Module, self).__init__(modules, in_modules, outputs)
        self._device = None
        self._address_book: Optional[AddressBook] = None
        self._modules_ordering: Optional[List[TorchModule]] = None
        self._fold_idx_info = fold_idx_info

    @property
    def has_address_book(self) -> bool:
        return self._address_book is not None

    @property
    def device(self) -> Optional[Union[str, torch.device, int]]:
        return self._device

    def topological_ordering(
        self, roots: Optional[Iterable[NodeType]] = None
    ) -> Iterator[NodeType]:
        if roots is None and self._modules_ordering is not None:
            return iter(self._modules_ordering)
        return super().topological_ordering(roots)

    def initialize_address_book(self) -> None:
        if self.has_address_book:
            raise RuntimeError("The address book has already been initialized")

        # Build the book address entries
        self._address_book = self._build_address_book()

        # Cache the topological ordering of the modules
        self._modules_ordering = list(super().topological_ordering())

    def to(
        self,
        device: Optional[Union[str, torch.device, int]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "TorchDiAcyclicGraph":
        if device is not None:
            self._address_book.set_device(device)
            self._device = device
        return cast(TorchDiAcyclicGraph, super().to(device, dtype, non_blocking))

    @abc.abstractmethod
    def _build_address_book(self) -> AddressBook:
        ...

    def _eval_forward(self, x: Optional[Tensor] = None) -> Tensor:
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        if not self.has_address_book:
            raise RuntimeError("The address book has not been initialized")
        module_outputs: List[Tensor] = []
        inputs_iterator = self._address_book.lookup(module_outputs, in_graph=x)
        for module, inputs in itertools.zip_longest(self._modules_ordering, inputs_iterator):
            if module is None:
                (output,) = inputs
                return output
            y = module(*inputs)
            module_outputs.append(y)
