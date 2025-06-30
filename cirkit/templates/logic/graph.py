import itertools
from abc import ABC
from collections import deque
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

    def __repr__(self) -> str:
        """Generate the repr string of the literal.

        Returns:
            str: The str representation of the node.
        """
        return str(self.literal)


class NegatedLiteralNode(LogicInputNode):
    """A negated literal in the logical circuit."""

    def __repr__(self) -> str:
        """Generate the repr string of the literal.

        Returns:
            str: The str representation of the node.
        """
        return f"¬ {self.literal}"


class ConjunctionNode(LogicCircuitNode):
    """A conjunction in the logical circuit."""


class DisjunctionNode(LogicCircuitNode):
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
        param = (
            np.array([[1.0, 0.0] * num_units]) if negated else np.array([[0.0, 1.0] * num_units])
        )
        param = param.reshape(num_units, 2)
        return CategoricalLayer(
            scope,
            num_categories=2,
            num_output_units=num_units,
            probs=Parameter.from_input(ConstantParameter(num_units, 2, value=param)),
        )

    return input_factory


class LogicCircuit(RootedDiAcyclicGraph[LogicCircuitNode]):
    def __init__(
        self,
        nodes: Sequence[LogicCircuitNode],
        in_nodes: dict[LogicCircuitNode, Sequence[LogicCircuitNode]],
        outputs: Sequence[LogicCircuitNode],
    ) -> None:
        """A Logic circuit represented as a rooted acyclic graph.

        Args:
            nodes (Sequence[LogicCircuitNode]): The list of nodes in the logic graph.
            in_nodes (dict[LogicCircuitNode, Sequence[LogicCircuitNode]]):
                A dictionary containing the list of inputs to each layer.
            outputs (Sequence[LogicCircuitNode]):
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
            if isinstance(node, (LogicInputNode, BottomNode, TopNode)):
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
    def inputs(self) -> Iterator[LogicCircuitNode]:
        """Returns the inputs of the circuit.

        Returns:
            Iterator[LogicCircuitNode]: Input of the circuit.
        """
        return (cast(LogicCircuitNode, node) for node in super().inputs)

    @property
    def outputs(self) -> Iterator[LogicCircuitNode]:
        """Returns the outputs of the circuit.

        Returns:
            Iterator[LogicCircuitNode]: Output of the circuit.
        """
        return (cast(LogicCircuitNode, node) for node in super().outputs)

    @property
    def literals(self) -> Iterator[LogicInputNode]:
        """Returns the literals in the graph.

        Returns:
            Iterator[LogicInputNode]: An iterator over all the literals in the graph.
        """
        return (
            cast(LogicCircuitNode, node) for node in self.inputs if isinstance(node, LogicInputNode)
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

    @property
    def disjunctions(self) -> Iterator[DisjunctionNode]:
        """Returns the disjunctions in the graph.

        Returns:
            Iterator[DisjunctionNode]: An iterator over the disjunctions in the graph.
        """
        return (node for node in self.nodes if isinstance(node, DisjunctionNode))

    @property
    def conjunctions(self) -> Iterator[ConjunctionNode]:
        """Returns the conjunctions in the graph.

        Returns:
            Iterator[ConjunctionNode]: An iterator over the conjunctions in the graph.
        """
        return (node for node in self.nodes if isinstance(node, ConjunctionNode))

    @cached_property
    def num_variables(self) -> int:
        """
        Returns the number of literals in the graph.

        Returns:
            int: The number of literals.
        """
        return len({i.literal for i in self.inputs if isinstance(i, LogicInputNode)})

    @cache
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
            case _:
                assert False, f"Unknown node type: {node.__class__}"

        return scope

    def smooth(self):
        """Convert the a logic circuit to a smooth logic circuit in place.
        see https://yoojungchoi.github.io/files/ProbCirc20.pdf and
        https://proceedings.neurips.cc/paper/2019/file/940392f5f32a7ade1cc201767cf83e31-Paper.pdf
        for more information.

        Returns:
            LogicCircuit: A new logic graph that is smooth.
        """
        # collect all the nodes of literals in the circuit
        literal_map: dict[tuple[int, bool], LogicCircuitNode] = {
            **{(l.literal, True): l for l in self.positive_literals},
            **{(l.literal, False): l for l in self.negated_literals},
        }

        # create smoothing disjunctions composed of a literal and its negation
        smoothing_map: dict[int, DisjunctionNode] = {}
        for l in self.node_scope(self.output):
            l_disjunction = DisjunctionNode()
            self._in_nodes[l_disjunction] = [
                literal_map.setdefault((l, True), LiteralNode(l)),
                literal_map.setdefault((l, False), NegatedLiteralNode(l)),
            ]
            smoothing_map[l] = l_disjunction

        disjunctions = list(self.disjunctions)
        for d in disjunctions:
            d_scope = self.node_scope(d)
            d_children = list(self.node_inputs(d))

            # if any child of the disjunction does not have the same scope of the
            # disjunction we replace it with a conjunction containing itself and
            # the smoorhing disjunctions needed to match the target scope
            for d_child in d_children:
                missing_literals = d_scope.difference(self.node_scope(d_child))

                if len(missing_literals) > 0:
                    if isinstance(d_child, ConjunctionNode):
                        # if d_child is already a conjunction we direcly add to it
                        smoothing_conjunction = d_child
                    else:
                        # create the conjunction node and place it between the node and its child
                        smoothing_conjunction = ConjunctionNode()
                        self._in_nodes[smoothing_conjunction] = [
                            d_child,
                        ]
                        self._in_nodes[d].remove(d_child)
                        self._in_nodes[d].append(smoothing_conjunction)
                        self._nodes.append(smoothing_conjunction)

                    for missing_literal in missing_literals:
                        smoothed_literal = smoothing_map[missing_literal]
                        self._in_nodes[smoothing_conjunction].append(smoothed_literal)

                        # register the smoothed literal in the graph and its children
                        # if needed
                        if smoothed_literal not in self._nodes:
                            self._nodes.append(smoothed_literal)
                            for child in self.node_inputs(smoothed_literal):
                                if child not in self._nodes:
                                    self._nodes.append(child)

        # filter out unused nodes
        self._in_nodes = {
            n: [i for i in n_inputs if i in self._nodes]
            for n, n_inputs in self._in_nodes.items()
            if n in self._nodes
        }

        # re-initialize the relevant parts of the graph
        self.__init__(self._nodes, self._in_nodes, self._outputs)

    def trim(self):
        """Prune a graph in place by applying unit propagation to conjunction and disjunctions.

        The resulting logic graph will not contain Top or Bottom nodes.
        See https://en.wikipedia.org/wiki/Unit_propagation.
        """
        # pruning is performed by visiting the graph bottom-up
        # if a node is a literal, we keep going
        # if it is a conjunction or a disjunction, we exclude null elements from its children
        # and replace it by its null element if one of its children is the absorbing element

        for node in filter(
            lambda n: isinstance(n, (ConjunctionNode, DisjunctionNode)), self.topological_ordering()
        ):
            absorbing_element, null_element = (
                (BottomNode, TopNode)
                if isinstance(node, ConjunctionNode)
                else (TopNode, BottomNode)
            )

            # remove null elements from child
            self._in_nodes[node] = [
                c for c in self._in_nodes[node] if not isinstance(c, null_element)
            ]

            # prune trivial node if absorbing element is within children
            if any(isinstance(c, absorbing_element) for c in self.node_inputs(node)):
                del self._in_nodes[node]

                # remove any reference to node
                self._in_nodes = {
                    n: [ni for ni in n_inputs if ni != node]
                    for n, n_inputs in self._in_nodes.items()
                }

        # re initialize the graph
        self.__init__(self._nodes, self._in_nodes, self._outputs)

    def compress(self):
        """The trimming operation might leave nodes unused.
        We can compress the graph by removing all the nodes that are not reachable
        from the root node."""
        on_the_path = set()
        visited = set()
        to_visit = deque(self.outputs)
        while to_visit:
            node = to_visit.popleft()
            visited.add(node)

            node_children = self.node_inputs(node)
            if node in self.literals:
                # literals are always accepted
                on_the_path.add(node)
            elif len(node_children) == 1:
                # if this node has only one child, then it is a trivial node
                # we can remove it and attach its parents as parents of the
                # unique children
                node_parents = self.node_outputs(node)

                if len(node_parents) == 0:
                    # we are replacing the root node with its child
                    self._outputs = [node_children[0]]
                    self._nodes.remove(node)
                else:
                    # the node has parents: connect them to its child
                    for node_parent in node_parents:
                        self._in_nodes[node_parent].remove(node)
                        self._in_nodes[node_parent].append(node_children[0])

                if node_children[0] not in visited:
                    to_visit.appendleft(node_children[0])

                # remove from nodes
                self._nodes = [n for n in self._nodes if n != node]
            elif len(node_children) > 1:
                on_the_path.add(node)

                # inspect children, if there are some that are
                # of the same type of this node, we can merge them
                # on this node and visit this node again
                for node_child in node_children:
                    if type(node) is type(node_child):
                        node_child_descendants = [
                            d for d in self.node_inputs(node_child) if d not in self._in_nodes[node]
                        ]

                        self._in_nodes[node].remove(node_child)
                        self._in_nodes[node].extend(node_child_descendants)

                to_visit.extendleft([c for c in node_children if c not in visited])

            # update graph metadata
            self._out_nodes = graph_nodes_outgoings(self._nodes, self.node_inputs)

        self._nodes = list(on_the_path)
        # filter out all nodes that have been compressed
        self._in_nodes = {
            n: [i for i in n_inputs if i in self._nodes]
            for n, n_inputs in self._in_nodes.items()
            if n in self._nodes
        }

        # re initialize the graph
        self.__init__(self._nodes, self._in_nodes, self._outputs)

        self._nodes = list(on_the_path)
        # filter out all nodes that have been compressed
        self._in_nodes = {
            n: [i for i in n_inputs if i in self._nodes]
            for n, n_inputs in self._in_nodes.items()
            if n in self._nodes
        }

        # re initialize the graph
        self.__init__(self._nodes, self._in_nodes, self._outputs)

    def build_circuit(
        self,
        literal_input_factory: InputLayerFactory = None,
        negated_literal_input_factory: InputLayerFactory = None,
        weight_factory: ParameterFactory | None = None,
        enforce_smoothness: bool = True,
        num_units: int = 1,
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
            num_units: Number of units. Defaults to 1 for deterministic circuit.

        Returns:
            Circuit: A symbolic circuit.
        """
        if (literal_input_factory == None) ^ (negated_literal_input_factory == None):
            raise ValueError(
                "Both literal_input_factory and negated_literal_input_factory should"
                "be specified at the same time or be none."
            )

        # remove bottom and top nodes by trimming the graph
        self.trim()

        # smooth the circuit if required
        if enforce_smoothness:
            self.smooth()

        # simplify the circuit by removing trivial and non-reachable nodes
        self.compress()

        in_layers: dict[Layer, Sequence[Layer]] = {}
        node_to_layer: dict[LogicCircuitNode, Layer] = {}

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
                    i_input = literal_input_factory(Scope([i.literal]), num_units=num_units)
                case NegatedLiteralNode():
                    i_input = negated_literal_input_factory(Scope([i.literal]), num_units=num_units)

            i_input.metadata["logic"]["source"] = i
            node_to_layer[i] = i_input

        for node in self.topological_ordering():
            match node:
                case ConjunctionNode():
                    product_node = HadamardLayer(num_units, arity=len(self.node_inputs(node)))
                    product_node.metadata["logic"]["source"] = node

                    in_layers[product_node] = [node_to_layer[i] for i in self.node_inputs(node)]
                    node_to_layer[node] = product_node
                case DisjunctionNode():
                    sum_node = SumLayer(
                        num_units,
                        1 if node == self.output else num_units,
                        arity=len(self.node_inputs(node)),
                        weight_factory=weight_factory,
                    )
                    sum_node.metadata["logic"]["source"] = node

                    in_layers[sum_node] = [node_to_layer[i] for i in self.node_inputs(node)]
                    node_to_layer[node] = sum_node

        layers = list(set(itertools.chain(*in_layers.values())).union(in_layers.keys()))
        return Circuit(layers, in_layers, [node_to_layer[self.output]])
