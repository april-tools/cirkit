import abc
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from torch import Tensor, nn

from cirkit.backend.torch.graph.nodes import TorchModule
from cirkit.utils.algorithms import DiAcyclicGraph


@dataclass(frozen=True)
class FoldIndexInfo:
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]]
    out_fold_idx: List[Tuple[int, int]]


@dataclass(frozen=True)
class AddressBookEntry:
    in_module_ids: List[List[int]]
    in_fold_idx: List[Optional[Tensor]]


TorchModuleType = TypeVar("TorchModuleType", bound=TorchModule)


class AddressBook(ABC):
    def __init__(
        self, entries: List[AddressBookEntry], *, in_network_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> None:
        self._entries = entries
        self._in_network_fn = in_network_fn

    def __len__(self) -> int:
        return len(self._entries)

    @abc.abstractmethod
    def lookup_module_inputs(
        self, module_id: int, module_outputs: List[Tensor], *, in_network: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        ...

    @abc.abstractmethod
    def lookup_output(self, module_outputs: List[Tensor]) -> Tensor:
        ...


class TorchDiAcyclicGraph(nn.Module, ABC, DiAcyclicGraph[TorchModuleType]):
    def __init__(
        self,
        modules: List[TorchModuleType],
        in_modules: Dict[TorchModuleType, List[TorchModuleType]],
        out_modules: Dict[TorchModuleType, List[TorchModuleType]],
        *,
        topologically_ordered: bool = False,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(ABC, self).__init__(
            modules, in_modules, out_modules, topologically_ordered=topologically_ordered
        )
        self._address_book: Optional[AddressBook] = None
        self._fold_idx_info = fold_idx_info

    @property
    def has_address_book(self) -> bool:
        return self._address_book is not None

    def initialize_address_book(self) -> None:
        if self.has_address_book:
            raise RuntimeError("The address book has already been initialized")
        fold_idx_info = self._fold_idx_info
        if fold_idx_info is None:
            fold_idx_info = build_fold_index_info(
                self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
            )
        self._address_book = self._build_address_book(fold_idx_info)
        self._fold_idx_info = None

    @abc.abstractmethod
    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        ...

    def _in_index(self, x: Tensor, idx: Tensor) -> Tensor:
        raise NotImplementedError()

    def _eval_forward(self, x: Optional[Tensor] = None) -> Tensor:
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        module_outputs: List[Tensor] = []
        for i, module in enumerate(self.topological_ordering()):
            inputs = self._address_book.lookup_module_inputs(i, module_outputs, in_network=x)
            y = module(*inputs)
            module_outputs.append(y)

        # Retrieve the output from the book address
        return self._address_book.lookup_output(module_outputs)


class TorchRootedDiAcyclicGraph(TorchDiAcyclicGraph[TorchModuleType]):
    @cached_property
    def output(self) -> TorchModuleType:
        outputs = list(self.outputs)
        if len(outputs) > 1:
            raise ValueError("The graph has more than one output node.")
        return outputs[0]


def build_fold_index_info(
    ordering: Iterable[TorchModuleType],
    *,
    outputs: Iterable[TorchModuleType],
    incomings_fn: Callable[[TorchModuleType], List[TorchModuleType]],
    in_address_fn: Optional[Callable[[TorchModuleType], List[int]]] = None,
) -> FoldIndexInfo:
    # A useful data structure mapping each unfolded module to
    # (i) a 'fold_id' (a natural number) pointing to the module layer it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the folded module,
    #      which recovers the output of the unfolded module.
    fold_idx: Dict[TorchModule, Tuple[int, int]] = {}

    # A useful data structure mapping each module id to
    # a tensor of indices IDX of size (F, H, 2), where F is the number of modules in the fold,
    # H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair (fold_id, slice_idx),
    # pointing to the folded module of id 'fold_id' and to the slice 'slice_idx' within that fold.
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]] = {}

    # Build the fold index information data structure, by following the topological ordering
    cur_module_id = 0
    for m in ordering:
        # Retrieve the input modules
        in_modules: List[TorchModule] = incomings_fn(m)

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
    ordering: Iterable[List[TorchModuleType]],
    *,
    outputs: Iterable[TorchModuleType],
    incomings_fn: Callable[[TorchModuleType], List[TorchModuleType]],
    group_foldable_fn: Callable[[List[TorchModuleType]], List[List[TorchModuleType]]],
    fold_group_fn: Callable[[List[TorchModuleType]], TorchModuleType],
    in_address_fn: Optional[Callable[[TorchModuleType], List[int]]] = None,
) -> Tuple[List[TorchModuleType], Dict[TorchModuleType, List[TorchModuleType]], FoldIndexInfo]:
    # A useful data structure mapping each unfolded module to
    # (i) a 'fold_id' (a natural number) pointing to the module layer it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the folded module,
    #      which recovers the output of the unfolded module.
    fold_idx: Dict[TorchModule, Tuple[int, int]] = {}

    # A useful data structure mapping each folded module id to
    # a tensor of indices IDX of size (F, H, 2), where F is the number of modules in the fold,
    # H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair (fold_id, slice_idx),
    # pointing to the folded module of id 'fold_id' and to the slice 'slice_idx' within that fold.
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]] = {}

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
            # Fold the modules group
            folded_module = fold_group_fn(group)

            # For each module in the group, retrieve the unfolded input modules
            in_group_modules: List[List[TorchModule]] = [incomings_fn(m) for m in group]

            # Set the input modules
            in_modules[folded_module] = list(
                set(modules[fold_idx[mi][0]] for msi in in_group_modules for mi in msi)
            )

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

    return modules, in_modules, FoldIndexInfo(in_fold_idx, out_fold_idx)
