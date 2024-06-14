from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, Dict, List, Iterable, Callable, Tuple, Optional

import numpy as np
import torch
from torch import nn, Tensor

from cirkit.utils.algorithms import Graph, DiAcyclicGraph, RootedDiAcyclicGraph

TorchModuleType = TypeVar('TorchModuleType', bound=nn.Module)


class TorchGraph(nn.Module, Graph[TorchModuleType]):
    def __init__(
        self,
        modules: List[TorchModuleType],
        in_modules: Dict[TorchModuleType, List[TorchModuleType]],
        out_modules: Dict[TorchModuleType, List[TorchModuleType]],
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(nn.Module, self).__init__(modules, in_modules, out_modules)


class TorchDiAcyclicGraph(nn.Module, DiAcyclicGraph[TorchModuleType]):
    def __init__(
        self,
        modules: List[TorchModuleType],
        in_modules: Dict[TorchModuleType, List[TorchModuleType]],
        out_modules: Dict[TorchModuleType, List[TorchModuleType]],
        *,
        topologically_ordered: bool = False
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(nn.Module, self).__init__(modules, in_modules, out_modules, topologically_ordered=topologically_ordered)


class TorchRootedDiAcyclicGraph(nn.Module, RootedDiAcyclicGraph[TorchModuleType]):
    def __init__(
        self,
        modules: List[TorchModuleType],
        in_modules: Dict[TorchModuleType, List[TorchModuleType]],
        out_modules: Dict[TorchModuleType, List[TorchModuleType]],
        *,
        topologically_ordered: bool = False
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(nn.Module, self).__init__(modules, in_modules, out_modules, topologically_ordered=topologically_ordered)

    @cached_property
    def output(self) -> TorchModuleType:
        outputs = list(self.outputs)
        if len(outputs) > 1:
            raise ValueError("The graph has more than one output node.")
        return outputs[0]


@dataclass(frozen=True)
class FoldAddressEntry:
    in_module_ids: List[List[int]]
    in_fold_idx: List[Optional[Tensor]]


FoldAddressBook = List[FoldAddressEntry]


def fold_graph(
    ordering: Iterable[List[TorchModuleType]],
    *,
    outputs: Iterable[TorchModuleType],
    incomings_fn: Callable[[TorchModuleType], List[TorchModuleType]],
    group_foldable_fn: Callable[[List[TorchModuleType]], List[List[TorchModuleType]]],
    fold_group_fn: Callable[[List[TorchModuleType]], TorchModuleType],
    in_fold_idx_fn: Optional[Callable[[List[TorchModuleType]], List[List[Tuple[int, int]]]]] = None
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
    fold_idx: Dict[TorchModuleType, Tuple[int, int]] = {}

    # A useful data structure mapping each folded module id to
    # a tensor of indices IDX of size (F, H, 2), where F is the number of modules in the fold,
    # H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair (fold_id, slice_idx),
    # pointing to the folded module of id 'fold_id' and to the slice 'slice_idx' within that fold.
    in_fold_idx: Dict[TorchModuleType, List[List[Tuple[int, int]]]] = {}

    # The list of folded modules and the inputs of each folded module
    modules: List[TorchModuleType] = []
    in_modules: Dict[TorchModuleType, List[TorchModuleType]] = {}

    # Fold modules in each frontier, by firstly finding the module groups to fold
    # in each frontier, and then by stacking each group of modules into a folded module
    for frontier in ordering:
        # Retrieve the module groups we can fold
        foldable_groups = group_foldable_fn(frontier)

        # Fold each group of modules
        for group in foldable_groups:
            # For each module in the group, retrieve the unfolded input modules
            in_group_modules: List[List[TorchModuleType]] = [incomings_fn(m) for m in group]

            # Check if we are folding input modules
            # If that is the case, we index some other input tensor, if specified.
            # If that is not the case, we retrieve the input index from one of the useful maps.
            in_modules_idx: List[List[Tuple[int, int]]]
            if in_group_modules[0]:
                in_modules_idx = [[fold_idx[mi] for mi in msi] for msi in in_group_modules]
            else:
                in_modules_idx = in_fold_idx_fn(group) if in_fold_idx_fn is not None else []

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


def build_unfolded_address_book(
    ordering: Iterable[TorchModuleType],
    *,
    incomings_fn: Callable[[TorchModuleType], List[TorchModuleType]],
    outputs: Iterable[TorchModuleType],
) -> FoldAddressBook:
    # The bookkeeping data structure
    book: FoldAddressBook = []

    # A map from module nodes to their ids
    module_ids: Dict[TorchModuleType, int] = {}

    # Build the bookkeeping data structure by following the topological ordering
    for m in ordering:
        # Retrieve the ids of the input modules
        in_module_ids = [[module_ids[m]] for m in incomings_fn(m)]

        # Build the bookkeeping entry
        # The modules are not folded, thus set None as the fold index
        entry = FoldAddressEntry(in_module_ids, [None] * len(in_module_ids))
        book.append(entry)

        # Set the module id
        module_ids[m] = len(module_ids)

    # Append a last bookkeeping entry, which contains the information to build the output tensor
    out_module_ids = list(map(module_ids.get, outputs))
    entry = FoldAddressEntry([out_module_ids], [None])
    book.append(entry)

    return book


def build_folded_address_book(
    ordering: Iterable[TorchModuleType],
    *,
    incomings_fn: Callable[[TorchModuleType], List[TorchModuleType]],
    num_folds_fn: Callable[[TorchModuleType], int],
    in_fold_idx: Dict[TorchModuleType, List[List[Tuple[int, int]]]],
    out_fold_idx: List[Tuple[int, int]],
    in_stack: bool = True
) -> FoldAddressBook:
    # The bookkeeping data structure
    book: FoldAddressBook = []

    # A useful dictionary mapping module ids to their number of folds
    num_folds: Dict[int, int] = {}

    # Build the bookkeeping data structure by following the topological ordering
    for m in ordering:
        # Retrieve the index information of the input modules
        in_modules_fold_idx = in_fold_idx[m]
        m_id = len(book)

        # Catch the case of a folded module without inputs
        if not in_modules_fold_idx:
            entry = FoldAddressEntry([], [])
        # Catch the case of a folded module having the input of the network as input
        elif len(incomings_fn(m)) == 0:
            input_idx = [[idx[1] for idx in fi] for fi in in_modules_fold_idx]
            input_idx_t = torch.tensor(input_idx)
            entry = FoldAddressEntry([], [input_idx_t])
        # Catch the case of a folded module having the output of another module as input
        else:
            entry = _build_folded_book_entry(
                in_modules_fold_idx, num_folds=num_folds, in_stack=in_stack
            )

        book.append(entry)
        num_folds[m_id] = num_folds_fn(m)
        continue

    # Append the last bookkeeping entry with the information to compute the output tensor
    entry = _build_folded_book_entry(
        [out_fold_idx], num_folds=num_folds, in_stack=True
    )
    book.append(entry)

    return book


def _build_folded_book_entry(
    in_fold_idx: List[List[Tuple[int, int]]],
    *,
    num_folds: Dict[int, int],
    in_stack: bool,
) -> FoldAddressEntry:
    # Build a folded book address entry
    if in_stack:
        # Retrieve the unique fold indices that reference the module inputs
        # We sort them for the purpose of easier debugging
        in_module_ids = sorted(list(set(idx[0] for fi in in_fold_idx for idx in fi)))

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

        return FoldAddressEntry([in_module_ids], [cum_fold_idx_t])

    # Transpose the index information
    in_fold_idx = list(map(list, zip(*in_fold_idx)))

    # The same as above, with in_stack=True, but repeating it for the input of the module
    # This is useful where inputs cannot be stacked in a single tensor,
    # e.g., for the parameter computational graph
    in_module_ids = [sorted(list(set(idx[0] for idx in fi))) for fi in in_fold_idx]
    cum_module_ids = [
        dict(zip(mids, np.cumsum([0] + [num_folds[mid] for mid in mids]).tolist()))
        for mids in in_module_ids
    ]
    cum_fold_idx_t: List[Optional[Tensor]] = []
    for i, fi in enumerate(in_fold_idx):
        cum_fold_i_idx: List[int] = [cum_module_ids[i][idx[0]] + idx[1] for idx in fi]
        cum_fold_i_idx_t: Optional[Tensor]
        if cum_fold_i_idx == list(range(len(cum_fold_i_idx))):
            cum_fold_i_idx_t = None
        else:
            cum_fold_i_idx_t = torch.tensor(cum_fold_i_idx)
        cum_fold_idx_t.append(cum_fold_i_idx_t)

    return FoldAddressEntry(in_module_ids, cum_fold_idx_t)
