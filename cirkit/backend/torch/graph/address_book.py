import abc
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class FoldIndexInfo:
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]]
    out_fold_idx: List[Tuple[int, int]]


@dataclass(frozen=True)
class AddressBookEntry:
    in_module_ids: List[List[int]]
    in_fold_idx: List[Optional[Tensor]]


class AddressBook(ABC):
    def __init__(self, entries: List[AddressBookEntry]) -> None:
        super().__init__()
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def set_device(self, device: Optional[Union[str, torch.device, int]] = None) -> "AddressBook":
        self._entries = list(
            map(
                lambda entry: AddressBookEntry(
                    entry.in_module_ids,
                    [idx if idx is None else idx.to(device) for idx in entry.in_fold_idx],
                ),
                self._entries,
            )
        )
        return self

    @abc.abstractmethod
    def lookup(
        self, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Iterator[Tuple[Tensor, ...]]:
        ...
