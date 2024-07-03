import abc
import itertools
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.graph.nodes import AbstractTorchModule, TorchModule


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
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    @abc.abstractmethod
    def lookup(
        self, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Iterator[Tuple[Tensor, ...]]:
        ...


def build_fold_index_info(
    ordering: Iterable[TorchModule],
    *,
    outputs: Iterable[TorchModule],
    incomings_fn: Callable[[TorchModule], List[TorchModule]],
    in_address_fn: Optional[Callable[[TorchModule], List[int]]] = None,
) -> FoldIndexInfo:
    # A useful data structure mapping each unfolded module to
    # (i) a 'fold_id' (a natural number) pointing to the module layer it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the folded module,
    #      which recovers the output of the unfolded module.
    fold_idx: Dict[AbstractTorchModule, Tuple[int, int]] = {}

    # A useful data structure mapping each module id to
    # a tensor of indices IDX of size (F, H, 2), where F is the number of modules in the fold,
    # H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair (fold_id, slice_idx),
    # pointing to the folded module of id 'fold_id' and to the slice 'slice_idx' within that fold.
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]] = {}

    # Build the fold index information data structure, by following the topological ordering
    cur_module_id = 0
    for m in ordering:
        # Retrieve the input modules
        in_modules: List[AbstractTorchModule] = incomings_fn(m)

        # Check if we are folding input modules
        # If that is the case, we index some other input tensor, if specified.
        # If that is not the case, we retrieve the input index from one of the useful maps.
        in_modules_idx: List[Tuple[int, int]]
        if in_modules:
            in_modules_idx = [fold_idx[mi] for mi in in_modules]
        elif in_address_fn is None:
            in_modules_idx = []
        else:
            in_modules_idx = [(-1, j) for j in in_address_fn(m)]

        # Update the data structures
        fold_idx[m] = (cur_module_id, 0)
        in_fold_idx[cur_module_id] = [in_modules_idx]
        cur_module_id += 1

    # Instantiate the information on how aggregate the outputs in a single tensor
    out_fold_idx = list(map(fold_idx.get, outputs))

    return FoldIndexInfo(in_fold_idx, out_fold_idx)


def build_folded_graph(
    ordering: Iterable[List[TorchModule]],
    *,
    outputs: Iterable[TorchModule],
    incomings_fn: Callable[[TorchModule], List[TorchModule]],
    fold_group_fn: Callable[[List[TorchModule]], TorchModule],
    in_address_fn: Optional[Callable[[TorchModule], List[int]]] = None,
) -> Tuple[
    List[TorchModule],
    Dict[TorchModule, List[TorchModule]],
    Dict[TorchModule, List[TorchModule]],
    FoldIndexInfo,
]:
    # A useful data structure mapping each unfolded module to
    # (i) a 'fold_id' (a natural number) pointing to the module layer it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the folded module,
    #      which recovers the output of the unfolded module.
    fold_idx: Dict[AbstractTorchModule, Tuple[int, int]] = {}

    # A useful data structure mapping each folded module id to
    # a tensor of indices IDX of size (F, H, 2), where F is the number of modules in the fold,
    # H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair (fold_id, slice_idx),
    # pointing to the folded module of id 'fold_id' and to the slice 'slice_idx' within that fold.
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]] = {}

    # The list of folded modules and the inputs/outputs of each folded module
    modules: List[AbstractTorchModule] = []
    in_modules: Dict[AbstractTorchModule, List[AbstractTorchModule]] = {}
    out_modules: Dict[AbstractTorchModule, List[TorchModule]] = defaultdict(list)

    # Fold modules in each frontier, by firstly finding the module groups to fold
    # in each frontier, and then by stacking each group of modules into a folded module
    for frontier in ordering:
        # Retrieve the module groups we can fold
        foldable_groups = group_foldable_modules(frontier)

        # Fold each group of modules
        for group in foldable_groups:
            # Fold the modules group
            folded_module = fold_group_fn(group)

            # For each module in the group, retrieve the unfolded input modules
            in_group_modules: List[List[AbstractTorchModule]] = [incomings_fn(m) for m in group]

            # Set the input modules
            folded_in_modules = list(
                set(modules[fold_idx[mi][0]] for msi in in_group_modules for mi in msi)
            )
            in_modules[folded_module] = folded_in_modules
            for mi in folded_in_modules:
                out_modules[mi].append(folded_module)

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

            # Update the data structures
            cur_module_id = len(modules)
            for i, m in enumerate(group):
                fold_idx[m] = (cur_module_id, i)
            in_fold_idx[cur_module_id] = in_modules_idx

            # Append the folded module
            modules.append(folded_module)

    # Instantiate the information on how aggregate the outputs in a single tensor
    out_fold_idx = list(map(fold_idx.get, outputs))

    return modules, in_modules, out_modules, FoldIndexInfo(in_fold_idx, out_fold_idx)


def group_foldable_modules(
    modules: List[TorchModule],
) -> List[List[TorchModule]]:
    # A dictionary mapping a module fold settings,
    # which uniquely identifies a group of modules that can be folded,
    # into a group of modules.
    groups: Dict[tuple, List[TorchModule]] = defaultdict(list)

    # For each module, either create a new group or insert it into an existing one
    for m in modules:
        groups[m.fold_settings].append(m)

    return list(groups.values())


def build_address_book_stacked_entry(
    in_fold_idx: List[List[Tuple[int, int]]], *, num_folds: Dict[int, int]
) -> AddressBookEntry:
    # Retrieve the unique fold indices that reference the module inputs
    in_module_ids = list(set(idx[0] for fi in in_fold_idx for idx in fi))

    # Compute the cumulative indices of the folded inputs
    cum_module_ids = dict(
        zip(
            in_module_ids,
            itertools.accumulate([0] + [num_folds[mid] for mid in in_module_ids]),
        )
    )

    # Build the bookkeeping entry
    cum_fold_idx = [[cum_module_ids[idx[0]] + idx[1] for idx in fi] for fi in in_fold_idx]
    useless_fold_idx = len(cum_fold_idx) == 1 and cum_fold_idx[0] == list(
        range(len(cum_fold_idx[0]))
    )
    cum_fold_idx_t = None if useless_fold_idx else torch.tensor(cum_fold_idx)

    return AddressBookEntry([in_module_ids], [cum_fold_idx_t])


def build_address_book_entry(
    in_fold_idx: List[List[Tuple[int, int]]], *, num_folds: Dict[int, int]
) -> AddressBookEntry:
    # Transpose the index information
    in_fold_idx = list(map(list, zip(*in_fold_idx)))

    # Retrieve the unique fold indices that reference the module inputs
    in_module_ids = [list(set(idx[0] for idx in hi)) for hi in in_fold_idx]

    # Compute the cumulative indices of the folded inputs
    cum_module_ids = [
        dict(zip(mids, itertools.accumulate([0] + [num_folds[mid] for mid in mids])))
        for mids in in_module_ids
    ]
    cum_fold_idx_t: List[Optional[Tensor]] = []
    for i, hi in enumerate(in_fold_idx):
        cum_fold_i_idx: List[int] = [cum_module_ids[i][idx[0]] + idx[1] for idx in hi]
        useless_fold_idx = cum_fold_i_idx == list(range(len(cum_fold_i_idx)))
        cum_fold_i_idx_t = None if useless_fold_idx else torch.tensor(cum_fold_i_idx)
        cum_fold_idx_t.append(cum_fold_i_idx_t)

    return AddressBookEntry(in_module_ids, cum_fold_idx_t)
