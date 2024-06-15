import abc
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Generic, Iterable, Callable, Tuple, Dict, TypeVar

import torch
from torch import Tensor

from cirkit.backend.torch.graph.nodes import TorchModule


AddressBookEntryType = TypeVar('AddressBookEntryType')


class AbstractAddressBook(ABC, Generic[AddressBookEntryType]):
    def __init__(
        self
    ) -> None:
        self._entries: List[AddressBookEntryType] = []

    @property
    def is_built(self) -> bool:
        return len(self._entries) > 0

    def __len__(self) -> int:
        return len(self._entries)

    @abc.abstractmethod
    def build(
        self,
        ordering: Iterable[TorchModule],
        *,
        outputs: Iterable[TorchModule],
        incomings_fn: Callable[[TorchModule], List[TorchModule]],
    ) -> None:
        ...

    @abc.abstractmethod
    def retrieve_module_inputs(
        self,
        module_id: int,
        module_outputs: List[Tensor],
        *,
        in_network_fn: Optional[Callable[[Tensor], Tensor]] = None
    ) -> Tuple[Tensor, ...]:
        ...

    @abc.abstractmethod
    def retrieve_output(self, module_outputs: List[Tensor]) -> Tensor:
        ...


@dataclass(frozen=True)
class AddressBookEntry:
    in_module_ids: List[int]
    in_idx: Optional[Tensor]


class AddressBook(AbstractAddressBook[AddressBookEntry]):
    def __init__(
        self,
        *,
        in_address_fn: Optional[Callable[[TorchModule], List[int]]] = None,
        stack_in_tensors: bool = False,
    ) -> None:
        super().__init__()
        self._in_address_fn = in_address_fn
        self._stack_in_tensors = stack_in_tensors

    def build(
        self,
        ordering: Iterable[TorchModule],
        *,
        outputs: Iterable[TorchModule],
        incomings_fn: Callable[[TorchModule], List[TorchModule]],
    ) -> None:
        if self.is_built:
            raise ValueError("The address book has already been built")

        # A map from module nodes to their ids
        module_ids: Dict[TorchModule, int] = {}

        # Build the address book data structure by following the topological ordering
        for m in ordering:
            # Build the address entry
            # Retrieve the ids of the input modules
            in_modules = incomings_fn(m)
            in_module_ids = [module_ids[mi] for mi in in_modules]

            # Build the address entry
            if in_modules:
                # The modules are not folded
                entry = AddressBookEntry(in_module_ids, None)
            # Catch the case of an input module not receiving the input of the computational graph
            elif self._in_address_fn is None:
                entry = AddressBookEntry([], None)
            # Catch the case of an input module receiving the input of the computational graph
            else:
                entry = AddressBookEntry([], torch.tensor([self._in_address_fn(m)]))

            # Set the module id
            module_ids[m] = len(module_ids)
            self._entries.append(entry)

        # Append a last bookkeeping entry, which contains the information to build the output tensor
        out_module_ids = list(map(module_ids.get, outputs))
        entry = AddressBookEntry(out_module_ids, None)
        self._entries.append(entry)

    def retrieve_module_inputs(
        self,
        module_id: int,
        module_outputs: List[Tensor],
        *,
        in_network_fn: Optional[Callable[[Tensor], Tensor]] = None
    ) -> Tuple[Tensor, ...]:
        if not self.is_built:
            raise ValueError("The address book has not been built")
        # Retrieve the input tensor given by other modules
        entry = self._entries[module_id]
        in_tensors = tuple(module_outputs[mid] for mid in entry.in_module_ids)

        # Catch the case there are no inputs coming from other modules
        if not in_tensors:
            # If an input tensor to the computational graph has been, then use it
            if in_network_fn is None:
                return ()
            assert entry.in_idx is not None
            x = in_network_fn(entry.in_idx)
            return x,

        # Catch the case there are some inputs coming from other modules
        if self._stack_in_tensors:
            x = torch.stack(in_tensors, dim=1)
            return x,
        return in_tensors

    def retrieve_output(self, module_outputs: List[Tensor]) -> Tensor:
        if not self.is_built:
            raise ValueError("The address book has not been built")
        outputs = self.retrieve_module_inputs(-1, module_outputs=module_outputs)
        assert len(outputs) == 1
        return outputs[0]
