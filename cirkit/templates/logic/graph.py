import itertools
from abc import ABC
from collections.abc import Iterator
from functools import cached_property
from typing import Sequence, cast, final

from cirkit.utils.algorithms import RootedDiAcyclicGraph
from cirkit.utils.scope import Scope


class LogicCircuitNode(ABC):
    """The abstract base class for nodes in logic circuits."""


class TopNode(LogicCircuitNode):
    """The top node representing True in the logic circuit."""


class BottomNode(LogicCircuitNode):
    """The bottom node representing False in the logic circuit."""


class LogicInputNode(LogicCircuitNode):
    """The abstract base class for input nodes in logic circuits."""

    def __init__(self, literal: int) -> None:
        """Init class.

        Args:
            literal (int): The literal of this node.
        """
        super().__init__()
        self._literal = literal

    @property
    def literal(self) -> int:
        """The literal of this input.

        Returns:
            str: The int representation of the literal.
        """
        return self._literal

    def __repr__(self) -> str:
        """Generate the repr string of the node.

        Returns:
            str: The str representation of the node.
        """
        return f"{type(self).__name__}@0x{id(self):x}({self.literal})"


class LiteralNode(LogicInputNode):
    """A literal in the logical circuit."""


class NegatedLiteralNode(LogicInputNode):
    """A negated literal in the logical circuit."""


class ConjunctionNode(LogicCircuitNode):
    """A conjunction in the logical circuit."""


class DisjunctionNode(LogicCircuitNode):
    """A conjunction in the logical circuit."""


class LogicGraph(RootedDiAcyclicGraph[LogicCircuitNode]):
    def __init__(
        self,
        nodes: Sequence[LogicCircuitNode],
        in_nodes: dict[LogicCircuitNode, Sequence[LogicCircuitNode]],
        outputs: Sequence[LogicCircuitNode],
    ) -> None:
        super().__init__(nodes, in_nodes, outputs)

    @property
    def inputs(self) -> Iterator[LogicCircuitNode]:
        return (cast(LogicCircuitNode, node) for node in super().inputs)

    @property
    def outputs(self) -> Iterator[LogicCircuitNode]:
        return (cast(LogicCircuitNode, node) for node in super().outputs)

    @cached_property
    def num_variables(self) -> int:
        return len({ i.literal for i in self.inputs if isinstance(i, LogicInputNode) })

    def node_scope(self, node: LogicCircuitNode) -> Scope:
        """Compute the scope of a node.

        Args:
            node (LogicCircuitNode): The node for which the scope is computed.

        Returns:
            Scope: The scope of the node.
        """
        match node:
            case TopNode() | BottomNode():
                scope = Scope([])
            case LiteralNode() | NegatedLiteralNode():
                scope = Scope([node.literal])
            case DisjunctionNode() | ConjunctionNode():
                scope = Scope([])
                for i in self.node_inputs(node):
                    scope = scope.union(self.node_scope(i))

        return scope

    def smooth(self) -> "LogicGraph":
        """Construct a new smooth graph from this current graph.
        see https://yoojungchoi.github.io/files/ProbCirc20.pdf and
        https://proceedings.neurips.cc/paper/2019/file/940392f5f32a7ade1cc201767cf83e31-Paper.pdf
        for more information.

        Returns:
            LogicGraph: A new logic graph that is smooth.
        """
        literal_map: dict[tuple[int, bool], LogicCircuitNode] = {
            (node.literal, isinstance(node, LiteralNode)): node
            for node in self.nodes
            if isinstance(node, (LiteralNode, NegatedLiteralNode))
        }
        # smoothing map keeps track of the disjunctions created for smoothing purposes
        smoothing_map: dict[int, DisjunctionNode] = {}
        disjunctions = filter(lambda x: isinstance(x, DisjunctionNode), self.nodes)

        in_nodes: dict[LogicCircuitNode, list[LogicCircuitNode]] = self._in_nodes.copy()

        for d in disjunctions:
            d_scope = self.node_scope(d)

            for conj in in_nodes[d]:
                missing_literals = d_scope.difference(self.node_scope(conj))

                if len(missing_literals) > 0:
                    for ml in missing_literals:
                        if ml not in smoothing_map:
                            ml_conjunction = ConjunctionNode()
                            in_nodes[ml_conjunction] = [
                                literal_map.get((ml, True), LiteralNode(ml)),
                                literal_map.get((ml, False), NegatedLiteralNode(ml)),
                            ]
                            smoothing_map[ml] = ml_conjunction
                        in_nodes[conj].append(smoothing_map[ml])

        nodes = set(itertools.chain(*in_nodes.values())).union(in_nodes.keys())
        return LogicGraph(nodes, in_nodes, self._outputs)
