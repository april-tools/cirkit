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
        if len(outputs) != 1:
            assert ValueError("A logic graphs can only have one output!")
        super().__init__(nodes, in_nodes, outputs)

    def simplify(self) -> "LogicGraph":
        """
        Simplify a graph by removed trivial nodes and propagating the result.

        Returns:
            LogicGraph: The simplified graph, where all bottom and top nodes have
                been removed through simplification. 
        """
        in_nodes = dict(self.nodes_inputs.copy())
        root = next(self.outputs)

        absorbing_element = lambda n: BottomNode if isinstance(n, ConjunctionNode) else TopNode
        null_element = lambda n: TopNode if isinstance(n, ConjunctionNode) else BottomNode
        
        absorbed_nodes = [
            n
            for n, children in in_nodes.items()
            if any([isinstance(child, absorbing_element(n)) for child in children])
        ]
        
        # update the graph
        in_nodes = {
            n: [child for child in children if not isinstance(child, null_element(n)) and child not in absorbed_nodes]
            for n, children in in_nodes.items()
            if n not in absorbed_nodes
        }

        nodes = list(set(itertools.chain(*in_nodes.values())).union(in_nodes.keys()))
        
        return LogicGraph(nodes=nodes, in_nodes=in_nodes, outputs=[root])

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

            for input_to_d in in_nodes[d]:
                to_add_for_smoothing: list[LogicCircuitNode] = []
                missing_literals = d_scope.difference(self.node_scope(input_to_d))

                if len(missing_literals) > 0:
                    for ml in missing_literals:
                        if ml not in smoothing_map:
                            # construct a conjunction representing the literal ml
                            # for smoothing purposes
                            smooth_ml = DisjunctionNode()
                            in_nodes[smooth_ml] = [
                                literal_map.get((ml, True), LiteralNode(ml)),
                                literal_map.get((ml, False), NegatedLiteralNode(ml)),
                            ]
                            smoothing_map[ml] = smooth_ml
                        
                        to_add_for_smoothing.append(smoothing_map[ml])
                
                    # if input to disjunction is a conjunction or a disjunction 
                    # then directly add to its inputs else create an ad-hoc node
                    if input_to_d in in_nodes:
                        in_nodes[input_to_d].extend(to_add_for_smoothing)
                    else:
                        ad_hoc = ConjunctionNode()
                        in_nodes[ad_hoc] = to_add_for_smoothing
                        in_nodes[ad_hoc].append(input_to_d)

                        # replace input_to_d with the ad-hoc disjunction
                        in_nodes[d].remove(input_to_d)
                        # add to the top so that it does not get checked again
                        in_nodes[d].insert(0, ad_hoc)

        nodes = set(itertools.chain(*in_nodes.values())).union(in_nodes.keys())
        return LogicGraph(nodes, in_nodes, self._outputs)
