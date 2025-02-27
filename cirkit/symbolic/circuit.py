import itertools
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from typing import Any

from cirkit.symbolic.layers import InputLayer, Layer, ProductLayer, SumLayer
from cirkit.utils.algorithms import (
    DiAcyclicGraph,
    RootedDiAcyclicGraph,
    bfs,
    subgraph,
    topological_ordering,
)
from cirkit.utils.scope import Scope


class StructuralPropertyError(Exception):
    """An exception that is raised when an error regarding one circuit structural property
    occurs."""

    def __init__(self, msg: str):
        """Initializes a structural property error with a message.

        Args:
            msg: The message.
        """
        super().__init__(msg)


@dataclass(frozen=True)
class StructuralProperties:
    """The available structural properties of a circuit."""

    smooth: bool
    """Whether the circuit is smooth."""
    decomposable: bool
    """Whether the circuit is decomposable."""
    structured_decomposable: bool
    """Whether the circuit is structured-decomposable, i.e., is compatible with itself."""
    omni_compatible: bool
    """Whether the circuit is omni-compatible, i.e., compatible to a fully-factorized circuit."""


class CircuitOperator(IntEnum):
    """The available symbolic operators defined over circuits."""

    CONCATENATE = auto()
    """The concatenation operator defined over many circuits."""
    EVIDENCE = auto()
    """The evidence operator defined over a circuit."""
    INTEGRATION = auto()
    """The integration operator defined over a circuit."""
    DIFFERENTIATION = auto()
    """The differentiation operator defined over a circuit."""
    MULTIPLICATION = auto()
    """The multiplication operator defined over two circuits."""
    CONJUGATION = auto()
    """The conjugatation operator defined over a circuit computing a complex function."""


@dataclass(frozen=True)
class CircuitOperation:
    """The symbolic operation that is applied to obtain a symbolic circuit."""

    operator: CircuitOperator
    """The circuit operator of the operation."""
    operands: tuple["Circuit", ...]
    """The circuit operands of the operation."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata of the operation."""


class CircuitBlock(RootedDiAcyclicGraph[Layer]):
    """The circuit block data structure. A circuit block is a fragment of a symbolic circuit,
    consisting of a single root (or output) layer. A circuit block can be of only two types:
    (1) a circuit block whose layers that do not have any inputs must all be input layers, and
    (2) a circuit block where there is one and only one layer that does not have any other input,
    which can be either a sum or product layer.
    """

    def __init__(
        self, layers: Sequence[Layer], in_layers: Mapping[Layer, list[Layer]], output: Layer
    ):
        """Initializes a circuit block.

        Args:
            layers: The sequence of layers in the block.
            in_layers: A dictionary containing the list of inputs to each layer.
            output: The root (or output) of the circuit block.
        """
        super().__init__(layers, in_layers, [output])

    def layer_inputs(self, sl: Layer) -> Sequence[Layer]:
        """Retrieves the inputs to a layer.

        Args:
            sl: The layer.

        Returns:
            Sequence[Layer]: The sequence of inputs.
        """
        return self.node_inputs(sl)

    def layer_outputs(self, sl: Layer) -> Sequence[Layer]:
        """Retrieves the outputs of a layer.

        Args:
            sl: The layer.

        Returns:
            Sequence[Layer]: The sequence of outputs.
        """
        return self.node_outputs(sl)

    @property
    def layers_inputs(self) -> Mapping[Layer, Sequence[Layer]]:
        """Retrieves the dictionary containing the sequence of inputs to each layer.

        Returns:
            Mapping[Layer, Sequence[Layer]]:
        """
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Mapping[Layer, Sequence[Layer]]:
        """Retrieves the dictionary containing the sequence of outputs of each layer.

        Returns:
            Mapping[Layer, Sequence[Layer]]:
        """
        return self.nodes_outputs

    @property
    def layers(self) -> Sequence[Layer]:
        """Retrieves a sequence of layers.

        Returns:
            Sequence[layer]:
        """
        return self.nodes

    @property
    def inner_layers(self) -> Iterator[SumLayer | ProductLayer]:
        """Retrieves an iterator over inner layers (i.e., layers that have at least one input).

        Returns:
            Iterator[Union[SumLayer, ProductLayer]]:
        """
        return (sl for sl in self.layers if isinstance(sl, (SumLayer, ProductLayer)))

    @property
    def sum_layers(self) -> Iterator[SumLayer]:
        """Retrieves an iterator over sum layers.

        Returns:
            Iterator[SumLayer]:
        """
        return (sl for sl in self.layers if isinstance(sl, SumLayer))

    @property
    def product_layers(self) -> Iterator[ProductLayer]:
        """Retrieves an iterator over product layers.

        Returns:
            Iterator[ProductLayer]:
        """
        return (sl for sl in self.layers if isinstance(sl, ProductLayer))

    @staticmethod
    def from_layer(sl: Layer) -> "CircuitBlock":
        """Instantiate a circuit block from a single layer.

        Args:
            sl: The layer.

        Returns:
            CircuitBlock: The circuit block consisting of only one layer.
        """
        return CircuitBlock([sl], {}, sl)

    @staticmethod
    def from_layer_composition(*layers: Layer) -> "CircuitBlock":
        """Instantiate a circuit block from a composition of multiple layers.
         The ordering of the composition is given by the ordering of the layers.

        Args:
            layers: A sequence of layers.

        Returns:
            CircuitBlock: The circuit block consisting of a composition of layers.

        Raises:
            ValueError: If the given sequence of layers consists of less than two layers.
        """
        layers = list(layers)
        in_layers = {}
        if len(layers) <= 1:
            raise ValueError("Expected a composition of at least 2 layers")
        for i, sl in enumerate(layers):
            in_layers[sl] = [layers[i - 1]] if i - 1 >= 0 else []
        return CircuitBlock(layers, in_layers, layers[-1])

    @staticmethod
    def from_nary_layer(lout: Layer, *ls: InputLayer) -> "CircuitBlock":
        """Instantiate a circuit block consisting of an output layer having
        multiple layers as inputs.

        Args:
            lout: The output layer.
            *ls: A sequence of inpput layers.

        Returns:
            CircuitBlock: The circuit block consisting of an output layer with several
                input layers as inputs.
        """
        layers = [lout, *ls]
        in_layers = {lout: list(ls)}
        return CircuitBlock(layers, in_layers, lout)


class Circuit(DiAcyclicGraph[Layer]):
    """The symbolic circuit representation."""

    def __init__(
        self,
        layers: Sequence[Layer],
        in_layers: Mapping[Layer, Sequence[Layer]],
        outputs: Sequence[Layer],
        *,
        operation: CircuitOperation | None = None,
    ) -> None:
        """Initializes a symbolic circuit.

        Args:
            layers: The list of symbolic layers.
            in_layers: A dictionary containing the list of inputs to each layer.
            outputs: The output layers of the circuit.
            operation: The optional operation the circuit has been obtained through.
        """
        super().__init__(layers, in_layers, outputs)
        self.operation = operation

        # Build scopes bottom-up, and check the consistency of the layers, w.r.t.
        # the arity and the number of input and output units
        self._scopes: dict[Layer, Scope] = {}
        for sl in self.topological_ordering():
            sl_ins = self.layer_inputs(sl)
            if isinstance(sl, InputLayer):
                self._scopes[sl] = sl.scope
                if len(sl_ins):
                    raise ValueError(
                        f"{sl}: found an input layer with {len(sl_ins)} layer inputs, "
                        "but expected none"
                    )
                continue
            self._scopes[sl] = Scope.union(*tuple(self._scopes[sli] for sli in sl_ins))
            if sl.arity != len(sl_ins):
                raise ValueError(
                    f"{sl}: expected arity {sl.arity}, " f"but found {len(sl_ins)} input layers"
                )
            sl_ins_units = [sli.num_output_units for sli in sl_ins]
            if any(sl.num_input_units != num_units for num_units in sl_ins_units):
                raise ValueError(
                    f"{sl}: expected number of input units {sl.num_input_units}, "
                    f"but found input layers {sl_ins}"
                )
        self.scope = Scope.union(*tuple(self._scopes[sl] for sl in self.outputs))

    @property
    def num_variables(self) -> int:
        """Retrieves the number of variables the circuit is defined on.

        Returns:
            int:
        """
        return len(self.scope)

    def layer_scope(self, sl: Layer) -> Scope:
        """Retrieves the scope of a layer.

        Args:
            sl: The layer.

        Returns:
            The scope of the given layer.
        """
        return self._scopes[sl]

    def layer_inputs(self, sl: Layer) -> Sequence[Layer]:
        """Retrieves the inputs to a layer.

        Args:
            sl: The layer.

        Returns:
            Sequence[Layer]: The list of inputs.
        """
        return self.node_inputs(sl)

    def layer_outputs(self, sl: Layer) -> Sequence[Layer]:
        """Retrieves the outputs of a layer.

        Args:
            sl: The layer.

        Returns:
            Sequence[Layer]: The list of outputs.
        """
        return self.node_outputs(sl)

    @property
    def layers_inputs(self) -> Mapping[Layer, Sequence[Layer]]:
        """Retrieves the dictionary containing the list of inputs to each layer.

        Returns:
            Dict[Layer, Sequence[Layer]]:
        """
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Mapping[Layer, Sequence[Layer]]:
        """Retrieves the dictionary containing the list of outputs of each layer.

        Returns:
            Dict[Layer, Sequence[Layer]]:
        """
        return self.nodes_outputs

    @property
    def layers(self) -> Sequence[Layer]:
        """Retrieves a sequence of layers.

        Returns:
            Sequence[layer]:
        """
        return self.nodes

    @property
    def inner_layers(self) -> Iterator[SumLayer | ProductLayer]:
        """Retrieves an iterator over inner layers (i.e., non-input layers).

        Returns:
            Iterator[Union[SumLayer, ProductLayer]]:
        """
        return (sl for sl in self.layers if isinstance(sl, (SumLayer, ProductLayer)))

    @property
    def input_layers(self) -> Iterator[InputLayer]:
        """Retrieves an iterator over input layers.

        Returns:
            Iterator[SumLayer]:
        """
        return (sl for sl in self.layers if isinstance(sl, InputLayer))

    @property
    def sum_layers(self) -> Iterator[SumLayer]:
        """Retrieves an iterator over sum layers.

        Returns:
            Iterator[SumLayer]:
        """
        return (sl for sl in self.layers if isinstance(sl, SumLayer))

    @property
    def product_layers(self) -> Iterator[ProductLayer]:
        """Retrieves an iterator over product layers.

        Returns:
            Iterator[ProductLayer]:
        """
        return (sl for sl in self.layers if isinstance(sl, ProductLayer))

    def subgraph(self, *outputs: Layer) -> "Circuit":
        """Retrieve the sub-circuit having the given layers as outputs.

        Args:
            *outputs: The output layers.

        Returns:
            The sub-circuit having the given layers as outputs.
        """
        layers, in_layers = subgraph(outputs, self.layer_inputs)
        return Circuit(layers, in_layers, outputs=outputs)

    ##################################### Structural properties ####################################

    @cached_property
    def is_smooth(self) -> bool:
        """Check if the circuit is smooth.

        Returns:
            bool: True if the circuit is smooth and False otherwise.
        """
        return all(
            self.layer_scope(sum_sl) == self.layer_scope(in_sl)
            for sum_sl in self.sum_layers
            for in_sl in self.layer_inputs(sum_sl)
        )

    @cached_property
    def is_decomposable(self) -> bool:
        """Check if the circuit is decomposable.

        Returns:
            bool: True if the circuit is decomposable and False otherwise.
        """
        return not any(
            self.layer_scope(in_sl1) & self.layer_scope(in_sl2)
            for prod_sl in self.product_layers
            for in_sl1, in_sl2 in itertools.combinations(self.layer_inputs(prod_sl), 2)
        )

    @cached_property
    def is_structured_decomposable(self) -> bool:
        """Check if the circuit is structured-decomposable.

        Returns:
            bool: True if the circuit is structured-decomposable and False otherwise.
        """
        if not self.is_smooth:
            return False
        if not self.is_decomposable:
            return False
        scope_factorizations = _scope_factorizations(self)
        return all(len(fs) == 1 for _, fs in scope_factorizations.items())

    @cached_property
    def is_omni_compatible(self) -> bool:
        """Check if the circuit is omni-compatible.

        Returns:
            bool: True if the circuit is omni-compatible and False otherwise.
        """
        if not self.is_smooth:
            return False
        if not self.is_decomposable:
            return False
        scope_factorizations = _scope_factorizations(self)
        vs = Scope(range(self.num_variables))
        return _are_compatible(scope_factorizations, {vs: {tuple(Scope([vid]) for vid in vs)}})

    @cached_property
    def properties(self) -> StructuralProperties:
        """Retrieves all the structural properties of the circuit: smoothness,
        decomposability, structured-decomposability and omni-compatibility.

        Returns:
            The structural properties.
        """
        return StructuralProperties(
            self.is_smooth,
            self.is_decomposable,
            self.is_structured_decomposable,
            self.is_omni_compatible,
        )

    @classmethod
    def from_operation(
        cls,
        blocks: list[CircuitBlock],
        in_blocks: dict[CircuitBlock, Sequence[CircuitBlock]],
        output_blocks: list[CircuitBlock],
        *,
        operation: CircuitOperation,
    ) -> "Circuit":
        """Constructs a circuit that resulted from an operation over other circuits.

        Args:
            blocks: The list of circuit blocks.
            in_blocks: A dictionary containing the list of block inputs to each circuit block.
            output_blocks: The outputs blocks of the circuit.
            operation: A circuit operation containing the information of the operation.

        Returns:
            Circuit: A symbolic circuit.Ki

        Raises:
            ValueError: If there is a circuit block having more than one layer with no inputs that
                are not input layers (i.e., they are either sum of product layers).
        """
        # Unwrap blocks into layers (as well as their connections)
        layers = [l for b in blocks for l in b.layers]
        in_layers = defaultdict(list)
        outputs = [b.output for b in output_blocks]

        # Retrieve connections between layers from connections between circuit blocks
        for b in blocks:
            b_layer_inputs = list(b.inputs)
            block_ins = in_blocks.get(b, [])
            if len(b_layer_inputs) == 1:
                (b_input,) = b_layer_inputs
                in_layers[b_input].extend(bi.output for bi in block_ins)
            elif len(block_ins) > 0:
                raise ValueError(
                    "A circuit block having multiple inputs cannot be a non-input block"
                )
            for sl in b.layers:
                in_layers[sl].extend(b.layer_inputs(sl))
        # Build the circuit and set the operation
        return cls(layers, in_layers, outputs, operation=operation)


def are_compatible(sc1: Circuit, sc2: Circuit) -> bool:
    """Check if two symbolic circuits are compatible.
     Note that compatibility is a commutative property of circuits.

    Args:
        sc1: The first symbolic circuit.
        sc2: The second symbolic circuit.

    Returns:
        bool: True if the first symbolic circuit is compatible with the second one.
    """
    if not sc1.is_smooth:
        return False
    if not sc1.is_decomposable:
        return False
    if not sc2.is_smooth:
        return False
    if not sc2.is_decomposable:
        return False
    sfs1 = _scope_factorizations(sc1)
    sfs2 = _scope_factorizations(sc2)
    return _are_compatible(sfs1, sfs2)


def pipeline_topological_ordering(roots: Sequence[Circuit]) -> Iterator[Circuit]:
    """Retrieves the topological ordering of circuits in a pipeline, given a sequence of
     root (or output) symbolic circuits in a pipeline.

    Args:
        roots: The sequence of root (or output) symbolic circuits in a pipeline.

    Returns:
        Iterator[Circuit]: An iterator of the topological ordering of circuits in a pipeline.
    """

    def _operands_fn(sc: Circuit) -> tuple[Circuit, ...]:
        return () if sc.operation is None else sc.operation.operands

    return topological_ordering(bfs(roots, incomings_fn=_operands_fn), incomings_fn=_operands_fn)


def _scope_factorizations(sc: Circuit) -> dict[Scope, set[tuple[Scope, ...]]]:
    # For each product layer, retrieves how it factorizes its scope
    scope_factorizations: dict[Scope, set[tuple[Scope, ...]]] = defaultdict(set)
    for sl in sc.product_layers:
        sl_scope = sc.layer_scope(sl)
        fs = tuple(sorted(sc.layer_scope(sli) for sli in sc.layer_inputs(sl)))
        # Remove empty scopes that appear in the factorization
        fs = tuple(s for s in fs if s)
        # Add it to the scope factorizations only if it is a factorization
        if len(fs) > 1:
            scope_factorizations[sl_scope].add(fs)
    return scope_factorizations


def _are_compatible(
    sfs1: dict[Scope, set[tuple[Scope, ...]]], sfs2: dict[Scope, set[tuple[Scope, ...]]]
) -> bool:
    # Check if two scope factorizations are compatible
    # TODO: how to allow for possible product layer rearrangements?
    for scope, fs1 in sfs1.items():
        fs2 = sfs2.get(scope, None)
        if fs2 is None:
            return False
        if len(fs1) != 1 or len(fs2) != 1:
            return False
        f1 = fs1.pop()
        f2 = fs2.pop()
        if f1 != f2:
            return False
    return True
