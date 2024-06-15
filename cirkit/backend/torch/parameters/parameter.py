from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from cirkit.backend.torch.graph.folding import (
    AddressBook,
    AddressBookEntry,
    FoldIndexInfo,
    build_address_book_entry,
    build_address_book_stacked_entry,
    build_fold_index_info,
)
from cirkit.backend.torch.graph.modules import TorchRootedDiAcyclicGraph
from cirkit.backend.torch.graph.nodes import TorchModule


class TorchParameterNode(TorchModule, ABC):
    """The abstract base class for all reparameterizations."""

    def __init__(self, *, num_folds: int = 1, **kwargs) -> None:
        """Init class."""
        super().__init__(num_folds=num_folds)

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return {}


class TorchParameterLeaf(TorchParameterNode, ABC):
    @property
    def is_initialized(self) -> bool:
        return True

    def initialize_(self) -> None:
        pass

    def __call__(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        ...


class TorchParameterOp(TorchParameterNode, ABC):
    def __init__(self, *in_shape: Tuple[int, ...], num_folds: int = 1):
        super().__init__(num_folds=num_folds)
        self.in_shapes = in_shape

    def __call__(self, *xs: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(*xs)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, *xs: Tensor) -> Tensor:
        ...


class ParameterAddressBook(AddressBook):
    def lookup_module_inputs(
        self, module_id: int, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        # Retrieve the input tensor given by other modules
        entry = self._entries[module_id]
        in_module_ids = entry.in_module_ids

        # Catch the case there are no inputs coming from other modules
        if not in_module_ids:
            return ()

        # Catch the case there are some inputs coming from other modules
        in_tensors = tuple(
            torch.cat(tuple(module_outputs[mid] for mid in mids), dim=0)
            for mids in entry.in_module_ids
        )
        x = tuple(
            t[in_idx] if in_idx is not None else t
            for t, in_idx in zip(in_tensors, entry.in_fold_idx)
        )
        return x

    def lookup_output(self, module_outputs: List[Tensor]) -> Tensor:
        outputs = self.lookup_module_inputs(-1, module_outputs=module_outputs)
        output = torch.cat(outputs, dim=0)
        (in_fold_idx,) = self._entries[-1].in_fold_idx
        if in_fold_idx is not None:
            output = output[in_fold_idx]
        return output

    @classmethod
    def from_index_info(
        cls, ordering: Iterable[TorchParameterNode], fold_idx_info: FoldIndexInfo
    ) -> "ParameterAddressBook":
        # The address book entries being built
        entries: List[AddressBookEntry] = []

        # A useful dictionary mapping module ids to their number of folds
        num_folds: Dict[int, int] = {}

        # Build the bookkeeping data structure by following the topological ordering
        for mid, m in enumerate(ordering):
            # Retrieve the index information of the input modules
            in_modules_fold_idx = fold_idx_info.in_fold_idx[mid]

            # Catch the case of a folded module having the input of the network as input
            if in_modules_fold_idx:
                entry = build_address_book_entry(in_modules_fold_idx, num_folds=num_folds)
            # Catch the case of a folded module without inputs
            else:
                entry = AddressBookEntry([], [])

            num_folds[mid] = m.num_folds
            entries.append(entry)

        # Append the last bookkeeping entry with the information to compute the output tensor
        entry = build_address_book_stacked_entry([fold_idx_info.out_fold_idx], num_folds=num_folds)
        entries.append(entry)

        return ParameterAddressBook(entries)


class TorchParameter(TorchRootedDiAcyclicGraph[TorchParameterNode]):
    @property
    def num_folds(self) -> int:
        return self.output.num_folds

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.output.shape

    def initialize_(self) -> None:
        """Reset the input parameters."""
        # TODO: assuming parameter operators do not have any learnable parameters
        for p in self.inputs:
            p.initialize_()

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        y = self._eval_forward()  # (F, d1, d2, ..., dk)
        return y

    def _build_address_book(self) -> AddressBook:
        fold_idx_info = self._fold_idx_info
        if fold_idx_info is None:
            fold_idx_info = build_fold_index_info(
                self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
            )
        address_book = ParameterAddressBook.from_index_info(
            self.topological_ordering(), fold_idx_info
        )
        self._fold_idx_info = None
        return address_book
