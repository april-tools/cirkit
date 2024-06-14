from functools import cached_property
from typing import TypeVar, Dict, List, Iterable, Callable, Tuple, Optional

from torch import nn

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


def fold_graph(
    ordering: Iterable[List[TorchModuleType]],
    *,
    incomings_fn: Callable[[TorchModuleType], List[TorchModuleType]],
    group_foldable_fn: Callable[[List[TorchModuleType]], List[List[TorchModuleType]]],
    fold_group_fn: Callable[[List[TorchModuleType]], TorchModuleType],
    in_fold_idx_fn: Optional[Callable[[List[TorchModuleType]], List[List[Tuple[int, int]]]]] = None
) -> Tuple[
    List[TorchModuleType],
    Dict[TorchModuleType, List[TorchModuleType]],
    Dict[TorchModuleType, Tuple[int, int]],
    Dict[TorchModuleType, List[List[Tuple[int, int]]]]
]:
    # A useful data structure mapping each unfolded module to
    # (i) a 'fold_id' (a natural number) pointing to the module layer it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the folded module,
    #      which recovers the output of the unfolded module.
    fold_idx: Dict[TorchModuleType, Tuple[int, int]] = {}

    # A useful data structure mapping each folded module to
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
            for i, l in enumerate(group):
                fold_idx[l] = (len(modules), i)
            modules.append(folded_module)
            in_fold_idx[folded_module] = in_modules_idx

    return modules, in_modules, fold_idx, in_fold_idx
