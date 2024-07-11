from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Type

from cirkit.backend.compiler import AbstractCompiler
from cirkit.backend.registry import CompilerRegistry
from cirkit.backend.torch.graph.optimize import GraphOptMatch, GraphOptPatternDefn
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameterNode

CircuitOptPatternDefn = GraphOptPatternDefn[TorchLayer]
CircuitOptPattern = Type[CircuitOptPatternDefn]
CircuitOptMatch = GraphOptMatch[TorchLayer]
CircuitOptApplyFunc = Callable[["TorchCompiler", CircuitOptMatch], TorchLayer]

ParameterOptPatternDefn = GraphOptPatternDefn[TorchParameterNode]
ParameterOptPattern = Type[ParameterOptPatternDefn]
ParameterOptMatch = GraphOptMatch[TorchParameterNode]
ParameterOptApplyFunc = Callable[["TorchCompiler", ParameterOptMatch], TorchParameterNode]


@dataclass(frozen=True)
class LayerOptPatternDefn:
    cls: Type[TorchLayer]
    patterns: Dict[str, ParameterOptPattern]


LayerOptPattern = Type[LayerOptPatternDefn]


@dataclass(frozen=True)
class LayerOptMatch:
    pattern: LayerOptPattern
    entry: TorchLayer
    matches: Dict[str, ParameterOptMatch]


LayerOptApplyFunc = Callable[["TorchCompiler", LayerOptMatch], Tuple[TorchLayer]]


class CircuitOptRegistry(CompilerRegistry[CircuitOptPattern, CircuitOptApplyFunc]):
    @classmethod
    def _validate_rule_signature(cls, func: CircuitOptApplyFunc) -> Optional[CircuitOptPattern]:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return None
        if not issubclass(args["compiler"], AbstractCompiler):
            return None
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_cls = args[arg_names[0]]
        if found_cls != CircuitOptMatch:
            return None
        return found_cls


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
