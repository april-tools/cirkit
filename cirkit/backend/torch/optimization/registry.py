from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Type

from cirkit.backend.compiler import AbstractCompiler
from cirkit.backend.registry import CompilerRegistry
from cirkit.backend.torch.graph.optimize import GraphOptMatch, GraphOptPatternDefn
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.leaves import TorchParameterNode

ParameterOptPatternDefn = GraphOptPatternDefn[TorchParameterNode]
ParameterOptPattern = Type[ParameterOptPatternDefn]
ParameterOptMatch = GraphOptMatch[TorchParameterNode]
ParameterOptApplyFunc = Callable[["TorchCompiler", ParameterOptMatch], TorchParameterNode]


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
        pentries: List[Dict[str, ParameterOptMatch]],
    ):
        super().__init__(pattern, entries)
        self._pentries = pentries

    @property
    def pentries(self) -> List[Dict[str, ParameterOptMatch]]:
        return self._pentries


LayerOptApplyFunc = Callable[["TorchCompiler", LayerOptMatch], Tuple[TorchLayer, ...]]


class ParameterOptRegistry(CompilerRegistry[ParameterOptPattern, ParameterOptApplyFunc]):
    @classmethod
    def _validate_rule_signature(cls, func: ParameterOptApplyFunc) -> Optional[ParameterOptPattern]:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return None
        if not issubclass(args["compiler"], AbstractCompiler):
            return None
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_cls = args[arg_names[0]]
        if found_cls != ParameterOptMatch:
            return None
        return found_cls


class LayerOptRegistry(CompilerRegistry[LayerOptPattern, LayerOptApplyFunc]):
    @classmethod
    def _validate_rule_signature(cls, func: LayerOptApplyFunc) -> Optional[LayerOptPattern]:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return None
        if not issubclass(args["compiler"], AbstractCompiler):
            return None
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_cls = args[arg_names[0]]
        if found_cls != LayerOptMatch:
            return None
        return found_cls
