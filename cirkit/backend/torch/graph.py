from functools import cached_property
from typing import TypeVar, Dict, List

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
