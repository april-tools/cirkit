from typing import TYPE_CHECKING, Protocol

from cirkit.backend.registry import CompilerRegistry
from cirkit.backend.torch.graph.optimize import GraphOptMatch, GraphOptPatternDefn
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.nodes import TorchParameterNode

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


ParameterOptPatternDefn = GraphOptPatternDefn[TorchParameterNode]
ParameterOptPattern = type[ParameterOptPatternDefn]
ParameterOptMatch = GraphOptMatch[TorchParameterNode]

LayerOptPatternDefn = GraphOptPatternDefn[TorchLayer]
LayerOptPattern = type[LayerOptPatternDefn]
LayerOptMatch = GraphOptMatch[TorchLayer]


class ParameterOptApplyFunc(Protocol):
    def __call__(
        self, compiler: "TorchCompiler", match: ParameterOptMatch
    ) -> tuple[TorchParameterNode, ...]: ...


class LayerOptApplyFunc(Protocol):
    def __call__(
        self, compiler: "TorchCompiler", match: LayerOptMatch
    ) -> tuple[TorchLayer, ...]: ...


class ParameterOptRegistry(CompilerRegistry[ParameterOptPattern, ParameterOptApplyFunc]):
    @classmethod
    def _validate_rule_function(cls, func: ParameterOptApplyFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return ann[args[-1]] == ParameterOptMatch


class LayerOptRegistry(CompilerRegistry[LayerOptPattern, LayerOptApplyFunc]):
    @classmethod
    def _validate_rule_function(cls, func: LayerOptApplyFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return ann[args[-1]] == LayerOptMatch
