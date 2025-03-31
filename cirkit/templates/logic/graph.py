import itertools
from abc import ABC
from collections.abc import Iterator, Sequence
from functools import cache, cached_property
from typing import cast

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import CategoricalLayer, HadamardLayer, InputLayer, Layer, SumLayer
from cirkit.symbolic.parameters import ConstantParameter, Parameter, ParameterFactory
from cirkit.templates.utils import InputLayerFactory
from cirkit.utils.algorithms import RootedDiAcyclicGraph, graph_nodes_outgoings
from cirkit.utils.scope import Scope


class LogicalCircuitNode(ABC):
    """The abstract base class for nodes in logic circuits."""


class TopNode(LogicalCircuitNode):
    """The top node representing True in the logic circuit."""


class BottomNode(LogicalCircuitNode):
    """The bottom node representing False in the logic circuit."""


class LogicalInputNode(LogicalCircuitNode):
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


class LiteralNode(LogicalInputNode):
    """A literal in the logical circuit."""

    def __repr__(self) -> str:
        """Generate the repr string of the literal.

        Returns:
            str: The str representation of the node.
        """
        return str(self.literal)


class NegatedLiteralNode(LogicalInputNode):
    """A negated literal in the logical circuit."""

    def __repr__(self) -> str:
        """Generate the repr string of the literal.

        Returns:
            str: The str representation of the node.
        """
        return f"¬ {self.literal}"


class ConjunctionNode(LogicalCircuitNode):
    """A conjunction in the logical circuit."""


class DisjunctionNode(LogicalCircuitNode):
    """A conjunction in the logical circuit."""


def default_literal_input_factory(negated: bool = False) -> InputLayerFactory:
    """Input factory for a boolean logic circuit input realized using a
    Categorical Layer constantly parametrized by a tensor [x, y] where x is
    the probability of being False and y the probability of being True.

    Args:
        negated (bool, optional): _description_. Defaults to False.

    Returns:
        InputLayerFactory: The input layer factory.
    """

    def input_factory(scope: Scope, num_units: int) -> InputLayer:
        """The default input factory maps literals to categorical distributions.
        Literals are parametrized by the probabilities [0.0, 1.0] while negated
        literals are parametrized by the probabilities [1.0, 0.0].

        Args:
            scope (Scope): Scope of the input corresponding to the literal id.
            num_units (int): Number of units in the input layer.

        Returns:
            InputLayer: Symbolic input layer.
        """
        param = np.array([[1.0, 0.0]]) if negated else np.array([[0.0, 1.0]])
        return CategoricalLayer(
            scope,
            num_categories=2,
            num_output_units=num_units,
            probs=Parameter.from_input(ConstantParameter(1, 2, value=param)),
        )

    return input_factory


class LogicalCircuit(RootedDiAcyclicGraph[LogicalCircuitNode]):
    def __init__(
        self,
        nodes: Sequence[LogicalCircuitNode],
        in_nodes: dict[LogicalCircuitNode, Sequence[LogicalCircuitNode]],
        outputs: Sequence[LogicalCircuitNode],
    ) -> None:
        """A Logical circuit represented as a rooted acyclic graph.

        Args:
            nodes (Sequence[LogicalCircuitNode]): The list of nodes in the logic graph.
            in_nodes (dict[LogicalCircuitNode, Sequence[LogicalCircuitNode]]):
                A dictionary containing the list of inputs to each layer.
            outputs (Sequence[LogicalCircuitNode]):
                The output layers of the circuit.
        """
        if len(outputs) != 1:
            assert ValueError("A logic graphs can only have one output!")
        super().__init__(nodes, in_nodes, outputs)

    def prune(self):
        """Prune the current graph by applying unit propagation.

        Prune a graph in place by applying unit propagation to conjunction and disjunctions.
        See https://en.wikipedia.org/wiki/Unit_propagation.
        """
        # pruning is performed by visiting the graph bottom-up
        # if a node is a literal, we keep going
        # if it is a conjunction or a disjunction, we exclude null elements from its children
        # and replace it by its null element if one of its children is the absorbing element
        absorbing_element = lambda n: BottomNode if isinstance(n, ConjunctionNode) else TopNode
        null_element = lambda n: TopNode if isinstance(n, ConjunctionNode) else BottomNode

        in_nodes = {}
        node_map = {n: n for n in self.nodes}
        for node in self.topological_ordering():
            if isinstance(node, (LogicalInputNode, BottomNode, TopNode)):
                pass
            elif isinstance(node, (ConjunctionNode, DisjunctionNode)):
                # gather current children excluding null elements
                children = [
                    node_map[c]
                    for c in self.node_inputs(node)
                    if not isinstance(node_map[c], null_element(node))
                ]

                # if one of the children is an absorbing element then
                # we replace this node with it
                if any(isinstance(c, absorbing_element(node)) for c in children):
                    node_map[node] = absorbing_element(node)()
                else:
                    in_nodes[node_map[node]] = children

        nodes = list(set(itertools.chain(*in_nodes.values())).union(in_nodes.keys()))

        # re initialize the graph
        self.__init__(nodes, in_nodes, list(self.outputs))

    def simplify(self):
        """Simplify the graph by removing orphan nodes."""
        # visit the graph top-down and remove nodes that are not in the subcircuit
        # identified by the root node
        in_nodes = {}

        to_visit = [self.output]
        while len(to_visit):
            node = to_visit.pop()

            children = in_nodes.get(node, [])
            if len(children) == 1 and node != self.output:
                # if the node has only one outgoing and one ingoing connections
                # then remove it and directly connect with child's descendants
                children = in_nodes.get(children[0], [])
                # to_visit.insert(0, node)

            if len(children) > 0:
                in_nodes[node] = children
                to_visit.extend(children)

        nodes = list(set(itertools.chain(*in_nodes.values())).union(in_nodes.keys()))

        # re initialize the graph
        self.__init__(nodes, in_nodes, list(self.outputs))

    @property
    def inputs(self) -> Iterator[LogicalCircuitNode]:
        """Returns the inputs of the circuit.

        Returns:
            Iterator[LogicalCircuitNode]: Input of the circuit.
        """
        return (cast(LogicalCircuitNode, node) for node in super().inputs)

    @property
    def outputs(self) -> Iterator[LogicalCircuitNode]:
        """Returns the outputs of the circuit.

        Returns:
            Iterator[LogicalCircuitNode]: Output of the circuit.
        """
        return (cast(LogicalCircuitNode, node) for node in super().outputs)

    @property
    def literals(self) -> Iterator[LogicalInputNode]:
        """Returns the literals in the graph.

        Returns:
            Iterator[LogicalInputNode]: An iterator over all the literals in the graph.
        """
        return (
            cast(LogicalCircuitNode, node)
            for node in self.inputs
            if isinstance(node, LogicalInputNode)
        )

    @property
    def positive_literals(self) -> Iterator[LiteralNode]:
        """Returns the literals in the graph excluding negated literals.

        Returns:
            Iterator[NegatedLiteralNode]: An iterator over the
                literals in the graph that are not negated.
        """
        return (node for node in self.literals if isinstance(node, LiteralNode))

    @property
    def negated_literals(self) -> Iterator[NegatedLiteralNode]:
        """Returns the negated literals in the graph.

        Returns:
            Iterator[NegatedLiteralNode]: An iterator over the negated
                literals in the graph.
        """
        return (node for node in self.literals if isinstance(node, NegatedLiteralNode))

    @cached_property
    def num_variables(self) -> int:
        """
        Returns the number of literals in the graph.

        Returns:
            int: The number of literals.
        """
        return len({i.literal for i in self.inputs if isinstance(i, LogicalInputNode)})

    @cache
    def node_scope(self, node: LogicalCircuitNode) -> Scope:
        """Compute the scope of a node.

        Args:
            node (LogicalCircuitNode): The node for which the scope is computed.

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
            case _:
                assert False, f"Unknown node type: {node.__class__}"

        return scope

    def smooth(self):
        """Convert the current graph to a smooth graph in place.
        see https://yoojungchoi.github.io/files/ProbCirc20.pdf and
        https://proceedings.neurips.cc/paper/2019/file/940392f5f32a7ade1cc201767cf83e31-Paper.pdf
        for more information.

        Returns:
            LogicalCircuit: A new logic graph that is smooth.
        """
        literal_map: dict[tuple[int, bool], LogicalCircuitNode] = {
            (node.literal, isinstance(node, LiteralNode)): node
            for node in self.nodes
            if isinstance(node, (LiteralNode, NegatedLiteralNode))
        }
        # smoothing map keeps track of the disjunctions created for smoothing purposes
        smoothing_map: dict[int, DisjunctionNode] = {}
        disjunctions = [n for n in self.nodes if isinstance(n, DisjunctionNode)]

        in_nodes = self._in_nodes
        for d in disjunctions:
            d_scope = self.node_scope(d)

            for input_to_d in self.node_inputs(d):
                to_add_for_smoothing: list[LogicalCircuitNode] = []
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
                    if not isinstance(input_to_d, LogicalInputNode):
                        in_nodes[input_to_d].extend(to_add_for_smoothing)
                    else:
                        ad_hoc = ConjunctionNode()
                        in_nodes[ad_hoc] = to_add_for_smoothing
                        in_nodes[ad_hoc].append(input_to_d)

                        # replace input_to_d with the ad-hoc disjunction
                        in_nodes[d].remove(input_to_d)
                        # add to the top so that it does not get checked again
                        in_nodes[d].insert(0, ad_hoc)

        nodes = list(set(itertools.chain(*in_nodes.values())).union(in_nodes.keys()))
        self.__init__(nodes, in_nodes, self._outputs)

    def build_circuit(
        self,
        literal_input_factory: InputLayerFactory = None,
        negated_literal_input_factory: InputLayerFactory = None,
        weight_factory: ParameterFactory | None = None,
        enforce_smoothness: bool = True,
    ) -> Circuit:
        """Construct a symbolic circuit from a logic circuit graph.
        If input factories for literals and their negation are not provided the it
        falls back to a categorical input layer with two categories parametrized by
        the constant vector [0, 1] for a literal and [1, 0] for its negation.

        Args:
            literal_input_factory: A factory that builds an input layer for a literal.
            negated_literal_input_factory: A factory that builds an input layer for a
                negated literal.
            weight_factory: The factory to construct the weight of sum layers.
                It can be None, or a parameter factory, i.e., a map from a shape to
                a symbolic parameter.
                If None is used, the default weight factory uses non-trainable unitary
                parameters, which instantiate a regular boolean logic graph.
            enforce_smoothness:
                Enforces smoothness of the circuit to support efficient marginalization.

        Returns:
            Circuit: A symbolic circuit.
        """
        if (literal_input_factory == None) ^ (negated_literal_input_factory == None):
            raise ValueError(
                "Both literal_input_factory and negated_literal_input_factory should"
                "be specified at the same time or be none."
            )
        if enforce_smoothness:
            self.smooth()
        self.prune()
        # self.simplify()

        in_layers: dict[Layer, Sequence[Layer]] = {}
        node_to_layer: dict[LogicalCircuitNode, Layer] = {}

        if (literal_input_factory is None) and (negated_literal_input_factory is None):
            literal_input_factory = default_literal_input_factory(negated=False)
            negated_literal_input_factory = default_literal_input_factory(negated=True)

        if weight_factory is None:
            # default to unitary weights
            def weight_factory(n: tuple[int]) -> Parameter:
                return Parameter.from_input(ConstantParameter(*n, value=1.0))

        # map each input literal to a symbolic input layer
        for i in self.inputs:
            match i:
                case LiteralNode():
                    i_input = literal_input_factory(Scope([i.literal]), num_units=1)
                case NegatedLiteralNode():
                    i_input = negated_literal_input_factory(Scope([i.literal]), num_units=1)

            i_input.metadata["logic"]["source"] = i
            node_to_layer[i] = i_input

        for node in self.topological_ordering():
            match node:
                case ConjunctionNode():
                    product_node = HadamardLayer(1, arity=len(self.node_inputs(node)))
                    product_node.metadata["logic"]["source"] = node

                    in_layers[product_node] = [node_to_layer[i] for i in self.node_inputs(node)]
                    node_to_layer[node] = product_node
                case DisjunctionNode():
                    sum_node = SumLayer(
                        1,
                        1,
                        arity=len(self.node_inputs(node)),
                        weight_factory=weight_factory,
                    )
                    sum_node.metadata["logic"]["source"] = node

                    in_layers[sum_node] = [node_to_layer[i] for i in self.node_inputs(node)]
                    node_to_layer[node] = sum_node

        layers = list(set(itertools.chain(*in_layers.values())).union(in_layers.keys()))
        return Circuit(layers, in_layers, [node_to_layer[self.output]])
