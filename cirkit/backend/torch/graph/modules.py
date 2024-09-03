from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union, cast

import torch
from torch import Tensor, nn

from cirkit.utils.algorithms import DiAcyclicGraph


class AbstractTorchModule(nn.Module, ABC):
    def __init__(self, *, num_folds: int = 1):
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def fold_settings(self) -> Tuple[Any, ...]:
        ...


TorchModule = TypeVar("TorchModule", bound=AbstractTorchModule)


@dataclass(frozen=True)
class FoldIndexInfo:
    ordering: List[TorchModule]
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]]
    out_fold_idx: List[Tuple[int, int]]


@dataclass(frozen=True)
class AddressBookEntry:
    module: Optional[TorchModule]
    in_module_ids: List[List[int]]
    in_fold_idx: List[Optional[Tensor]]


class AddressBook(ABC):
    def __init__(self, entries: List[AddressBookEntry]) -> None:
        super().__init__()
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[AddressBookEntry]:
        return iter(self._entries)

    def set_device(self, device: Optional[Union[str, torch.device, int]] = None) -> "AddressBook":
        self._entries = list(
            map(
                lambda entry: AddressBookEntry(
                    entry.module,
                    entry.in_module_ids,
                    [idx if idx is None else idx.to(device) for idx in entry.in_fold_idx],
                ),
                self._entries,
            )
        )
        return self

    @abstractmethod
    def lookup(
        self, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Iterator[Tuple[Optional[TorchModule], Tuple[Tensor, ...]]]:
        ...


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
        if fold_idx_info is None:
            fold_idx_info = self._build_unfold_index_info()
        self._address_book = self._build_address_book(fold_idx_info)

    @property
    def device(self) -> Optional[Union[str, torch.device, int]]:
        return self._device

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

    def _eval_forward(self, x: Optional[Tensor] = None) -> Tensor:
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        module_outputs: List[Tensor] = []
        for module, inputs in self._address_book.lookup(module_outputs, in_graph=x):
            if module is None:
                (output,) = inputs
                return output
            y = module(*inputs)
            module_outputs.append(y)

    @abstractmethod
    def _build_unfold_index_info(self) -> FoldIndexInfo:
        ...

    @abstractmethod
    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        ...

    def __repr__(self) -> str:
        def indent(s: str) -> str:
            s = s.split("\n")
            r = s[0]
            if len(s) == 1:
                return r
            return r + "\n" + "\n".join(f"  {t}" for t in s[1:])

        lines = [self.__class__.__name__ + "("]
        extra_lines = self.extra_repr()
        if extra_lines:
            lines.append(f"  {indent(extra_lines)}")
        for i, entry in enumerate(self._address_book):
            if entry.module is None:
                continue
            repr_module = indent(repr(entry.module))
            lines.append(f"  ({i}): {repr_module}")
        lines.append(")")
        return "\n".join(lines)
