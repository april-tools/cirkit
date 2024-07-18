from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Protocol, Tuple, Type

from cirkit.backend.compiler import AbstractCompiler
from cirkit.backend.registry import CompilerRegistry
from cirkit.backend.torch.graph.optimize import GraphOptMatch, GraphOptPatternDefn
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.leaves import TorchParameterNode

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


ParameterOptPatternDefn = GraphOptPatternDefn[TorchParameterNode]
ParameterOptPattern = Type[ParameterOptPatternDefn]
ParameterOptMatch = GraphOptMatch[TorchParameterNode]


class ParameterOptApplyFunc(Protocol):
    def __call__(
        self, compiler: "TorchCompiler", match: ParameterOptMatch
    ) -> Tuple[TorchLayer, ...]:
        ...


class LayerOptPatternDefn(GraphOptPatternDefn[TorchLayer]):
    @classmethod
    def entries(cls) -> List[Type[TorchLayer]]:
        ...

    @classmethod
    def ppatterns(cls) -> List[Dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]


LayerOptPattern = Type[LayerOptPatternDefn]


class LayerOptMatch(GraphOptMatch[TorchLayer]):
    def __init__(
        self,
        pattern: LayerOptPattern,
        entries: List[TorchLayer],
        pentries: List[Dict[str, List[ParameterOptMatch]]],
    ):
        super().__init__(pattern, entries)
        self._pentries = pentries

    @property
    def pentries(self) -> List[Dict[str, List[ParameterOptMatch]]]:
        return self._pentries

    @cached_property
    def size(self) -> int:
        size = super().size
        for pentry in self._pentries:
            for pmatches in pentry.values():
                size += sum(match.size for match in pmatches)
        return size


class LayerOptApplyFunc(Protocol):
    def __call__(self, compiler: "TorchCompiler", match: LayerOptMatch) -> Tuple[TorchLayer, ...]:
        ...


class ParameterOptRegistry(CompilerRegistry[ParameterOptPattern, ParameterOptApplyFunc]):
    @classmethod
    def _validate_rule_function(cls, func: ParameterOptApplyFunc) -> bool:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return False
        if not issubclass(args["compiler"], AbstractCompiler):
            return False
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_cls = args[arg_names[0]]
        return found_cls == ParameterOptMatch


class LayerOptRegistry(CompilerRegistry[LayerOptPattern, LayerOptApplyFunc]):
    @classmethod
    def _validate_rule_function(cls, func: LayerOptApplyFunc) -> bool:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return False
        if not issubclass(args["compiler"], AbstractCompiler):
            return False
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_cls = args[arg_names[0]]
        return found_cls == LayerOptMatch
