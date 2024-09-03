from copy import copy as shallowcopy
from functools import cached_property
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.graph.folding import (
    build_address_book_entry,
    build_address_book_stacked_entry,
    build_unfold_index_info,
)
from cirkit.backend.torch.graph.modules import (
    AddressBook,
    AddressBookEntry,
    FoldIndexInfo,
    TorchDiAcyclicGraph,
)
from cirkit.backend.torch.parameters.nodes import (
    TorchParameterNode,
    TorchPointerParameter,
    TorchTensorParameter,
)
from cirkit.utils.algorithms import topologically_process_nodes


class ParameterAddressBook(AddressBook):
    def lookup(
        self, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Iterator[Tuple[Optional[TorchParameterNode], Tuple[Tensor, ...]]]:
        # A useful function combining the modules outputs, and then possibly applying an index
        def select_index(mids: List[int], idx: Optional[Tensor]) -> Tensor:
            if len(mids) == 1:
                t = module_outputs[mids[0]]
            else:
                t = torch.cat([module_outputs[mid] for mid in mids], dim=0)
            return t if idx is None else t[idx]

        # Loop through the entries and yield inputs
        for entry in self._entries:
            in_module_ids = entry.in_module_ids

            # Catch the case there are some inputs coming from other modules
            if in_module_ids:
                x = tuple(
                    select_index(mids, in_idx)
                    for mids, in_idx in zip(in_module_ids, entry.in_fold_idx)
                )
                yield entry.module, x
                continue

            # Catch the case there are no inputs coming from other modules
            yield entry.module, ()

    @classmethod
    def from_index_info(cls, fold_idx_info: FoldIndexInfo) -> "ParameterAddressBook":
        # The address book entries being built
        entries: List[AddressBookEntry] = []

        # A useful dictionary mapping module ids to their number of folds
        num_folds: Dict[int, int] = {}

        # Build the bookkeeping data structure by following the topological ordering
        for mid, m in enumerate(fold_idx_info.ordering):
            # Retrieve the index information of the input modules
            in_modules_fold_idx = fold_idx_info.in_fold_idx[mid]

            # Catch the case of a folded module having the input of the network as input
            if in_modules_fold_idx:
                entry = build_address_book_entry(m, in_modules_fold_idx, num_folds=num_folds)
            # Catch the case of a folded module without inputs
            else:
                entry = AddressBookEntry(m, [], [])

            num_folds[mid] = m.num_folds
            entries.append(entry)

        # Append the last bookkeeping entry with the information to compute the output tensor
        entry = build_address_book_stacked_entry(
            None, [fold_idx_info.out_fold_idx], num_folds=num_folds, output=True
        )
        entries.append(entry)

        return ParameterAddressBook(entries)


class TorchParameter(TorchDiAcyclicGraph[TorchParameterNode]):
    @cached_property
    def num_folds(self) -> int:
        return sum(n.num_folds for n in self.outputs)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return next(self.outputs).shape

    def extract_subgraphs(self, *roots: TorchParameterNode) -> List["TorchParameter"]:
        # The set of torch tensor nodes being observed
        nodes_ptensor = set()

        def replace_ref_or_copy(n: TorchParameterNode) -> TorchParameterNode:
            if isinstance(n, TorchTensorParameter):
                if n in nodes_ptensor:
                    return TorchPointerParameter(n)
                nodes_ptensor.add(n)
            return shallowcopy(n)

        # Extract parameter sub-computational graphs that are rooted by the provided roots
        # If the sub-computational graphs would share torch tensor parameters (that are not pointers),
        # then parameter sharing is ensured by introducing torch parameter pointers, based on the
        # order specified by the roots.
        pgraphs = []
        for r in roots:
            nodes, in_nodes, outputs = topologically_process_nodes(
                self.topological_ordering(roots=[r]),
                [r],
                replace_ref_or_copy,
                incomings_fn=self.node_inputs,
            )
            pgraph = TorchParameter(nodes, in_nodes, outputs)
            pgraphs.append(pgraph)
        return pgraphs

    def _build_unfold_index_info(self) -> FoldIndexInfo:
        return build_unfold_index_info(
            self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
        )

    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        return ParameterAddressBook.from_index_info(fold_idx_info)

    def reset_parameters(self) -> None:
        """Reset the input parameters."""
        for p in self.nodes:
            p.reset_parameters()

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        return self._eval_forward()  # (F, d1, d2, ..., dk)
