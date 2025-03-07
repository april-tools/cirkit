import itertools
from abc import ABC
from collections.abc import Iterator, Sequence
from functools import cache, cached_property
from typing import cast

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import CategoricalLayer, HadamardLayer, InputLayer, Layer, SumLayer
from cirkit.symbolic.parameters import Parameter, ParameterFactory, TensorParameter
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
        param = np.array([1.0, 0.0]) if negated else np.array([0.0, 1.0])
        initializer = ConstantTensorInitializer(param)
        return CategoricalLayer(
            scope,
            num_categories=2,
            num_output_units=num_units,
            probs=Parameter.from_input(TensorParameter(1, 2, initializer=initializer)),
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
        Nodes that are not used as input to other nodes and are not among the output nodes
        are removed too.
        """
        absorbing_element = lambda n: BottomNode if isinstance(n, ConjunctionNode) else TopNode
        null_element = lambda n: TopNode if isinstance(n, ConjunctionNode) else BottomNode

        @cache
        def absorb_node(node):
            if isinstance(node, (ConjunctionNode, DisjunctionNode)):
                children = [absorb_node(c) for c in self.node_inputs(node)]

                # if the node contains the absorbing element, then it is replaced
                # altogether
                if any(isinstance(c, absorbing_element(node)) for c in children):
                    return absorbing_element(node)()

            return node

        # apply node absorbion and remove null elements from conjunctions and disjunctions
        in_nodes = {}
        for n, children in self._in_nodes.items():
            absorbed = absorb_node(n)

            if not isinstance(absorbed, (TopNode, BottomNode)):
                in_nodes[n] = [
                    c
                    for c in [absorb_node(c) for c in children]
                    if not isinstance(c, null_element(n))
                ]

        # remove nodes that are not used as input to any other node if they are not the output node
        out_nodes = graph_nodes_outgoings(self.nodes, lambda n: in_nodes.get(n, []))
        in_nodes = {
            n: children
            for n, children in in_nodes.items()
            if len(out_nodes.get(n, [])) > 0 or n in self._outputs
        }

        nodes = list(set(itertools.chain(*in_nodes.values())).union(in_nodes.keys()))

        # re initialize the graph
        self.__init__(nodes, in_nodes, list(self.outputs))

    def simplify(self):
        """Promote nodes that only have one child"""

        in_nodes = self._in_nodes
        output = self.output
        for node in self.topological_ordering():
            if len(self.node_inputs(node)) == 1:
                # replace all occurrences of this node with child
                in_nodes = {
                    n: [self.node_inputs(node)[0] if c == node else c for c in children]
                    for n, children in in_nodes.items()
                    if n != node
                }

                if node == output:
                    output = self.node_inputs(node)[0]

        nodes = list(set(itertools.chain(*in_nodes.values())).union(in_nodes.keys()))

        # re initialize the graph
        self.__init__(nodes, in_nodes, [output])

    @property
    def inputs(self) -> Iterator[LogicalCircuitNode]:
        return (cast(LogicalCircuitNode, node) for node in super().inputs)

    @property
    def outputs(self) -> Iterator[LogicalCircuitNode]:
        return (cast(LogicalCircuitNode, node) for node in super().outputs)

    @property
    def literals(self):
        return (
            cast(LogicalCircuitNode, node) for node in self.inputs if isinstance(node, LiteralNode)
        )

    @property
    def positive_literals(self):
        return (
            cast(LogicalCircuitNode, node)
            for node in self.literals
            if isinstance(node, LiteralNode)
        )

    @property
    def negated_literals(self):
        return (
            cast(LogicalCircuitNode, node)
            for node in self.literals
            if isinstance(node, NegatedLiteralNode)
        )

    @cached_property
    def num_variables(self) -> int:
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
        self.simplify()

        in_layers: dict[Layer, Sequence[Layer]] = {}
        node_to_layer: dict[LogicalCircuitNode, Layer] = {}

        if (literal_input_factory is None) and (negated_literal_input_factory is None):
            literal_input_factory = default_literal_input_factory(negated=False)
            negated_literal_input_factory = default_literal_input_factory(negated=True)

        if weight_factory is None:
            # default to unitary weights
            def weight_factory(n: tuple[int]) -> Parameter:
                # locally import numpy to avoid dependency on the whole file
                initializer = ConstantTensorInitializer(1.0)
                return Parameter.from_input(TensorParameter(*n, initializer=initializer))

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
