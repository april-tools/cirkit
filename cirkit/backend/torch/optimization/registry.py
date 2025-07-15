from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol

from cirkit.backend.registry import CompilerRegistry
from cirkit.backend.torch.graph.optimize import GraphOptMatch, GraphOptPatternDefn
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.nodes import TorchParameterNode

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


ParameterOptPatternDefn = GraphOptPatternDefn[TorchParameterNode]
ParameterOptPattern = type[ParameterOptPatternDefn]
ParameterOptMatch = GraphOptMatch[TorchParameterNode]


class ParameterOptApplyFunc(Protocol):
    def __call__(
        self, compiler: "TorchCompiler", match: ParameterOptMatch
    ) -> tuple[TorchParameterNode, ...]: ...


class LayerOptPatternDefn(GraphOptPatternDefn[TorchLayer]):
    @classmethod
    def entries(cls) -> list[type[TorchLayer]]:
        return NotImplemented

    @classmethod
    def ppatterns(cls) -> list[dict[str, ParameterOptPattern]]:
        return NotImplemented

    @classmethod
    def cpatterns(cls) -> list[dict[str, Any]]:
        return NotImplemented


LayerOptPattern = type[LayerOptPatternDefn]


class LayerOptMatch(GraphOptMatch[TorchLayer]):
    def __init__(
        self,
        pattern: LayerOptPattern,
        entries: Sequence[TorchLayer],
        pentries: Sequence[Mapping[str, list[ParameterOptMatch]]],
    ):
        super().__init__(pattern, entries)
        self._pentries = pentries

    @property
    def pentries(self) -> Sequence[Mapping[str, list[ParameterOptMatch]]]:
        return self._pentries

    @cached_property
    def size(self) -> int:
        size = super().size
        for pentry in self._pentries:
            for pmatches in pentry.values():
                size += sum(match.size for match in pmatches)
        return size


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
