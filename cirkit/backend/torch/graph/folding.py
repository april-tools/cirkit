from dataclasses import dataclass
from typing import List, Optional, Iterable, Callable, Tuple, Dict

import numpy as np
import torch
from torch import Tensor

from cirkit.backend.torch.graph.books import AbstractAddressBook
from cirkit.backend.torch.graph.modules import TorchModuleType
from cirkit.backend.torch.graph.nodes import TorchModule


@dataclass(frozen=True)
class FoldAddressBookEntry:
    in_module_ids: List[List[int]]
    in_fold_idx: List[Optional[Tensor]]


class FoldAddressBook(AbstractAddressBook[FoldAddressBookEntry]):
    def __init__(
        self,
        *,
        stack_in_tensors: bool = False,
        in_fold_idx: Dict[TorchModule, List[List[Tuple[int, int]]]],
        out_fold_idx: List[Tuple[int, int]]
    ):
        super().__init__()
        self._stack_in_tensors = stack_in_tensors
        self._in_fold_idx = in_fold_idx
        self._out_fold_idx = out_fold_idx

    def build(
        self,
        ordering: Iterable[TorchModule],
        *,
        outputs: Iterable[TorchModule],
        incomings_fn: Callable[[TorchModule], List[TorchModule]],
    ) -> None:
        # A useful dictionary mapping module ids to their number of folds
        num_folds: Dict[int, int] = {}

        # Build the bookkeeping data structure by following the topological ordering
        for m in ordering:
            # Retrieve the index information of the input modules
            in_modules_fold_idx = self._in_fold_idx[m]

            # Catch the case of a folded module without inputs
            if not in_modules_fold_idx:
                entry = FoldAddressBookEntry([], [])
            # Catch the case of a folded module having the input of the network as input
            elif len(incomings_fn(m)) == 0:
                input_idx = [[idx[1] for idx in fi] for fi in in_modules_fold_idx]
                input_idx_t = torch.tensor(input_idx)
                entry = FoldAddressBookEntry([], [input_idx_t])
            # Catch the case of a folded module having the output of another module as input
            else:
                entry = self._build_folded_book_entry(
                    in_modules_fold_idx, num_folds=num_folds,
                    stack_in_tensors=self._stack_in_tensors
                )

            m_id = self.__len__()
            num_folds[m_id] = m.num_folds
            self._entries.append(entry)

        # Append the last bookkeeping entry with the information to compute the output tensor
        entry = self._build_folded_book_entry(
            [self._out_fold_idx], num_folds=num_folds, stack_in_tensors=True
        )
        self._entries.append(entry)

    @staticmethod
    def _build_folded_book_entry(
        in_fold_idx: List[List[Tuple[int, int]]],
        num_folds: Dict[int, int],
        *,
        stack_in_tensors: bool = False,
    ) -> FoldAddressBookEntry:
        # Build a folded book address entry
        if stack_in_tensors:
            # Retrieve the unique fold indices that reference the module inputs
            # We sort them for the purpose of easier debugging
            in_module_ids = list(set(idx[0] for fi in in_fold_idx for idx in fi))

            # Compute the cumulative indices of the folded inputs
            cum_module_ids = dict(
                zip(in_module_ids, np.cumsum([0] + [num_folds[mid] for mid in in_module_ids]).tolist())
            )

            # Build the bookkeeping entry
            cum_fold_idx = [[cum_module_ids[idx[0]] + idx[1] for idx in fi] for fi in in_fold_idx]
            cum_fold_idx_t: Optional[Tensor]
            if len(cum_fold_idx) == 1 and cum_fold_idx[0] == list(range(len(cum_fold_idx[0]))):
                cum_fold_idx_t = None
            else:
                cum_fold_idx_t = torch.tensor(cum_fold_idx)

            return FoldAddressBookEntry([in_module_ids], [cum_fold_idx_t])

        # Transpose the index information
        transposed_in_fold_idx = list(map(list, zip(*in_fold_idx)))

        # The same as above, with in_stack=True, but repeating it for the input of the module
        # This is useful where inputs cannot be stacked in a single tensor,
        # e.g., for the parameter computational graph
        in_module_ids = [list(set(idx[0] for idx in hi)) for hi in transposed_in_fold_idx]
        cum_module_ids = [
            dict(zip(mids, np.cumsum([0] + [num_folds[mid] for mid in mids]).tolist()))
            for mids in in_module_ids
        ]
        cum_fold_idx_t: List[Optional[Tensor]] = []
        for i, hi in enumerate(transposed_in_fold_idx):
            cum_fold_i_idx: List[int] = [cum_module_ids[i][idx[0]] + idx[1] for idx in hi]
            cum_fold_i_idx_t: Optional[Tensor]
            if cum_fold_i_idx == list(range(len(cum_fold_i_idx))):
                cum_fold_i_idx_t = None
            else:
                cum_fold_i_idx_t = torch.tensor(cum_fold_i_idx)
            cum_fold_idx_t.append(cum_fold_i_idx_t)

        return FoldAddressBookEntry(in_module_ids, cum_fold_idx_t)

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
        in_tensors = tuple(
            tuple(module_outputs[mid] for mid in mids)
            for mids in entry.in_module_ids
        )

        # Catch the case there are no inputs coming from other modules
        if not in_tensors:
            # If an input tensor to the computational graph has been, then use it
            if in_network_fn is None:
                return ()
            assert entry.in_fold_idx is not None
            in_fold_idx, = entry.in_fold_idx
            x = in_network_fn(in_fold_idx)
            return x,

        # Catch the case there are some inputs coming from other modules
        if self._stack_in_tensors:
            in_tensors, = in_tensors
            in_fold_idx, = entry.in_fold_idx
            x = torch.cat(in_tensors, dim=0)
            x = x.unsqueeze(dim=0) if in_fold_idx is None else x[in_fold_idx]
            return x,
        in_tensors = tuple(torch.cat(ts, dim=0) for ts in in_tensors)
        in_tensors = tuple(
            t[in_idx] if in_idx is not None else t
            for t, in_idx in zip(in_tensors, entry.in_fold_idx)
        )
        return in_tensors

    def retrieve_output(self, module_outputs: List[Tensor]) -> Tensor:
        if not self.is_built:
            raise ValueError("The address book has not been built")
        outputs = self.retrieve_module_inputs(-1, module_outputs=module_outputs)
        output = torch.cat(outputs, dim=0)
        in_fold_idx, = self._entries[-1].in_fold_idx
        if in_fold_idx is not None:
            output = output[in_fold_idx]
        return output


def fold_graph(
    ordering: Iterable[List[TorchModuleType]],
    *,
    outputs: Iterable[TorchModuleType],
    incomings_fn: Callable[[TorchModuleType], List[TorchModuleType]],
    group_foldable_fn: Callable[[List[TorchModuleType]], List[List[TorchModuleType]]],
    fold_group_fn: Callable[[List[TorchModuleType]], TorchModuleType],
    in_address_fn: Optional[Callable[[TorchModuleType], List[int]]] = None
) -> Tuple[
    List[TorchModuleType],
    Dict[TorchModuleType, List[TorchModuleType]],
    Dict[TorchModuleType, List[List[Tuple[int, int]]]],
    List[Tuple[int, int]]
]:
    # A useful data structure mapping each unfolded module to
    # (i) a 'fold_id' (a natural number) pointing to the module layer it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the folded module,
    #      which recovers the output of the unfolded module.
    fold_idx: Dict[TorchModule, Tuple[int, int]] = {}

    # A useful data structure mapping each folded module id to
    # a tensor of indices IDX of size (F, H, 2), where F is the number of modules in the fold,
    # H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair (fold_id, slice_idx),
    # pointing to the folded module of id 'fold_id' and to the slice 'slice_idx' within that fold.
    in_fold_idx: Dict[TorchModule, List[List[Tuple[int, int]]]] = {}

    # The list of folded modules and the inputs of each folded module
    modules: List[TorchModule] = []
    in_modules: Dict[TorchModule, List[TorchModule]] = {}

    # Fold modules in each frontier, by firstly finding the module groups to fold
    # in each frontier, and then by stacking each group of modules into a folded module
    for frontier in ordering:
        # Retrieve the module groups we can fold
        foldable_groups = group_foldable_fn(frontier)

        # Fold each group of modules
        for group in foldable_groups:
            # For each module in the group, retrieve the unfolded input modules
            in_group_modules: List[List[TorchModule]] = [incomings_fn(m) for m in group]

            # Check if we are folding input modules
            # If that is the case, we index some other input tensor, if specified.
            # If that is not the case, we retrieve the input index from one of the useful maps.
            in_modules_idx: List[List[Tuple[int, int]]]
            if in_group_modules[0]:
                in_modules_idx = [[fold_idx[mi] for mi in msi] for msi in in_group_modules]
            elif in_address_fn is None:
                in_modules_idx = []
            else:
                in_modules_idx = [[(-1, j) for j in in_address_fn(m)] for m in group]

            # Fold the modules group
            folded_module = fold_group_fn(group)

            # Set the input modules
            in_modules[folded_module] = list(
                set(modules[fold_idx[mi][0]] for msi in in_group_modules for mi in msi)
            )

            # Update the data structures
            fold_id = len(modules)
            for i, l in enumerate(group):
                fold_idx[l] = (fold_id, i)
            in_fold_idx[folded_module] = in_modules_idx

            # Append the folded module
            modules.append(folded_module)

    # Instantiate the information on how aggregate the outputs in a single tensor
    out_fold_idx = list(map(fold_idx.get, outputs))

    return modules, in_modules, in_fold_idx, out_fold_idx
