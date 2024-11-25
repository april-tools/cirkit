import itertools
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from typing import Any, Protocol, cast

from cirkit.symbolic.layers import (
    HadamardLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import ParameterFactory
from cirkit.symbolic.parameters import (
    ConstantParameter,
    Parameter,
    ParameterFactory,
    TensorParameter,
)
from cirkit.templates.logic import (
    BottomNode,
    ConjunctionNode,
    DisjunctionNode,
    LiteralNode,
    LogicCircuitNode,
    LogicGraph,
    NegatedLiteralNode,
    TopNode,
    default_literal_input_factory,
)
from cirkit.templates.region_graph import PartitionNode, RegionGraph, RegionGraphNode, RegionNode
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

    def __init__(self, layers: Sequence[Layer], in_layers: dict[Layer, list[Layer]], output: Layer):
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
    def layers_inputs(self) -> dict[Layer, Sequence[Layer]]:
        """Retrieves the dictionary containing the sequence of inputs to each layer.

        Returns:
            Dict[Layer, Sequence[Layer]]:
        """
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> dict[Layer, Sequence[Layer]]:
        """Retrieves the dictionary containing the sequence of outputs of each layer.

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


class InputLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs input layers."""

    def __call__(self, scope: Scope, num_units: int, num_channels: int) -> InputLayer:
        """Constructs an input layer.

        Args:
            scope: The scope of the layer.
            num_units: The number of input units composing the layer.
            num_channels: The number of channel variables.

        Returns:
            InputLayer: An input layer.
        """


class SumLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs sum layers."""

    def __call__(self, num_input_units: int, num_output_units: int) -> SumLayer:
        """Constructs a sum layer.

        Args:
            num_input_units: The number of units in each layer that is an input.
            num_output_units: The number of sum units in the layer.

        Returns:
            SumLayer: A sum layer.
        """


class ProductLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs product layers."""

    def __call__(self, num_input_units: int, arity: int) -> ProductLayer:
        """Constructs a product layer.

        Args:
            num_input_units: The number of units in each layer that is an input.
            arity: The number of input layers.

        Returns:
            ProductLayer: A product layer.
        """


class Circuit(DiAcyclicGraph[Layer]):
    """The symbolic circuit representation."""

    def __init__(
        self,
        num_channels: int,
        layers: Sequence[Layer],
        in_layers: dict[Layer, Sequence[Layer]],
        outputs: Sequence[Layer],
        *,
        operation: CircuitOperation | None = None,
    ) -> None:
        """Initializes a symbolic circuit.

        Args:
            num_channels: The number of channels for each variable.
            layers: The list of symbolic layers.
            in_layers: A dictionary containing the list of inputs to each layer.
            outputs: The output layers of the circuit.
            operation: The optional operation the circuit has been obtained through.
        """
        super().__init__(layers, in_layers, outputs)
        self.num_channels = num_channels
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
        return len(self.scope) * self.num_channels

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
    def layers_inputs(self) -> dict[Layer, Sequence[Layer]]:
        """Retrieves the dictionary containing the list of inputs to each layer.

        Returns:
            Dict[Layer, Sequence[Layer]]:
        """
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> dict[Layer, Sequence[Layer]]:
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

    def subgraph(self, *roots: Layer) -> "Circuit":
        layers, in_layers = subgraph(roots, self.layer_inputs)
        return type(self)(self.num_channels, layers, in_layers, outputs=roots)

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
        num_channels: int,
        blocks: list[CircuitBlock],
        in_blocks: dict[CircuitBlock, Sequence[CircuitBlock]],
        output_blocks: list[CircuitBlock],
        *,
        operation: CircuitOperation,
    ) -> "Circuit":
        """Constructs a circuit that resulted from an operation over other circuits.

        Args:
            num_channels: The number of channels per variable.
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
        return cls(num_channels, layers, in_layers, outputs, operation=operation)

    @classmethod
    def from_region_graph(
        cls,
        region_graph: RegionGraph,
        *,
        input_factory: InputLayerFactory,
        sum_product: str | None = None,
        sum_weight_factory: ParameterFactory | None = None,
        nary_sum_weight_factory: ParameterFactory | None = None,
        sum_factory: SumLayerFactory | None = None,
        prod_factory: ProductLayerFactory | None = None,
        num_channels: int = 1,
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
        factorize_multivariate: bool = True,
    ) -> "Circuit":
        """Construct a symbolic circuit from a region graph.
            There are two ways to use this method. The first one is to specify a sum-product layer
            abstraction, which can be either 'cp' (the CP layer), 'cp-t' (the CP-transposed layer),
            or 'tucker' (the Tucker layer). The second one is to manually specify the factories to
            build distinct um and product layers. If the first way is chosen, then one can possibly
            use a factory that builds the symbolic parameters of the sum-product layer abstractions.
            The factory that constructs the input factory must always be specified.

        Args:
            region_graph: The region graph.
            input_factory: A factory that builds an input layer.
            sum_product: The sum-product layer to use. It can be None, 'cp', 'cp-t', or 'tucker'.
            sum_weight_factory: The factory to construct the weights of the sum layers.
                It can be None, or a parameter factory, i.e., a map
                from a shape to a symbolic parameter. If it is None, then the default
                weight factory of the sum layer is used instead.
            nary_sum_weight_factory: The factory to construct the weight of sum layers havity arity
                greater than one. If it is None, then it will have the same value and semantics of
                the given sum_weight_factory.
            sum_factory: A factory that builds a sum layer. It can be None.
            prod_factory: A factory that builds a product layer. It can be None.
            num_channels: The number of channels for each variable.
            num_input_units: The number of input units.
            num_sum_units: The number of sum units per sum layer.
            num_classes: The number of output classes.
            factorize_multivariate: Whether to fully factorize input layers, when they depend on
                more than one variable.

        Returns:
            Circuit: A symbolic circuit.

        Raises:
            NotImplementedError: If an unknown 'sum_product' is given.
            ValueError: If both 'sum_product' and layer factories are specified, or none of them.
            ValueError: If 'sum_product' is specified, but 'weight_factory' is not.
            ValueError: The given region graph is malformed.
        """
        if (sum_factory is None and prod_factory is not None) or (
            sum_factory is not None and prod_factory is None
        ):
            raise ValueError(
                "Both 'sum_factory' and 'prod_factory' must be specified or none of them"
            )
        if sum_product is None and (sum_factory is None or prod_factory is None):
            raise ValueError(
                "If 'sum_product' is not given, then both 'sum_factory' and 'prod_factory'"
                " must be specified"
            )
        if sum_product is not None and (sum_factory is not None or prod_factory is not None):
            raise ValueError(
                "At most one between 'sum_product' and the pair 'sum_factory' and 'prod_factory'"
                " must be specified"
            )
        if nary_sum_weight_factory is None:
            nary_sum_weight_factory = sum_weight_factory

        layers: list[Layer] = []
        in_layers: dict[Layer, list[Layer]] = {}
        node_to_layer: dict[RegionGraphNode, Layer] = {}

        def build_cp_(
            rgn: RegionNode, rgn_partitioning: Sequence[RegionNode]
        ) -> HadamardLayer | SumLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            denses = [
                SumLayer(
                    node_to_layer[rgn_in].num_output_units,
                    num_sum_units,
                    weight_factory=sum_weight_factory,
                )
                for rgn_in in rgn_partitioning
            ]
            hadamard = HadamardLayer(num_sum_units, arity=len(rgn_partitioning))
            layers.extend(denses)
            layers.append(hadamard)
            in_layers[hadamard] = denses
            for d, li in zip(denses, layer_ins):
                in_layers[d] = [li]
            # If the region is not a root region of the region graph,
            # then make Hadamard the last layer
            if region_graph.region_outputs(rgn):
                node_to_layer[rgn] = hadamard
                return hadamard
            # Otherwise, introduce an additional sum layer to ensure the output layer is a sum
            output_dense = SumLayer(
                hadamard.num_output_units, num_classes, weight_factory=sum_weight_factory
            )
            layers.append(output_dense)
            in_layers[output_dense] = [hadamard]
            node_to_layer[rgn] = output_dense
            return output_dense

        def build_cp_transposed_(
            rgn: RegionNode, rgn_partitioning: Sequence[RegionNode]
        ) -> SumLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list({li.num_output_units for li in layer_ins})
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a CP transposed layer, as the inputs would have different units"
                )
            num_units = num_sum_units if region_graph.region_outputs(rgn) else num_classes
            hadamard = HadamardLayer(num_in_units[0], arity=len(rgn_partitioning))
            dense = SumLayer(num_in_units[0], num_units, weight_factory=sum_weight_factory)
            layers.append(hadamard)
            layers.append(dense)
            in_layers[hadamard] = layer_ins
            in_layers[dense] = [hadamard]
            node_to_layer[rgn] = dense
            return dense

        def build_tucker_(rgn: RegionNode, rgn_partitioning: Sequence[RegionNode]) -> SumLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list({li.num_output_units for li in layer_ins})
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a Tucker layer, as the inputs would have different units"
                )
            num_units = num_sum_units if region_graph.region_outputs(rgn) else num_classes
            kronecker = KroneckerLayer(num_in_units[0], arity=len(rgn_partitioning))
            dense = SumLayer(
                kronecker.num_output_units,
                num_units,
                weight_factory=sum_weight_factory,
            )
            layers.append(kronecker)
            layers.append(dense)
            in_layers[kronecker] = layer_ins
            in_layers[dense] = [kronecker]
            node_to_layer[rgn] = dense
            return dense

        # Set the sum-product layer builder, if necessary
        sum_prod_builder_: Callable[[RegionNode, Sequence[RegionNode]], Layer] | None
        if sum_product is None:
            sum_prod_builder_ = None
        elif sum_product == "cp":
            sum_prod_builder_ = build_cp_
        elif sum_product == "cp-t":
            sum_prod_builder_ = build_cp_transposed_
        elif sum_product == "tucker":
            sum_prod_builder_ = build_tucker_
        else:
            raise NotImplementedError(f"Unknown sum-product layer abstraction called {sum_product}")

        # Loop through the region graph nodes, which are already sorted in a topological ordering
        for node in region_graph.topological_ordering():
            if isinstance(node, PartitionNode):  # Partition node
                # If a sum-product layer abstraction has been specified,
                # then just skip partition nodes
                if sum_prod_builder_ is not None:
                    continue
                assert prod_factory is not None
                partition_inputs = region_graph.partition_inputs(node)
                prod_inputs = [node_to_layer[rgn] for rgn in partition_inputs]
                prod_sl = prod_factory(num_sum_units, len(prod_inputs))
                layers.append(prod_sl)
                in_layers[prod_sl] = prod_inputs
                node_to_layer[node] = prod_sl
            assert isinstance(
                node, RegionNode
            ), "Region graph nodes must be either region or partition nodes"
            region_inputs = region_graph.region_inputs(node)
            region_outputs = region_graph.region_outputs(node)
            if not region_inputs:
                # Input region node
                if factorize_multivariate and len(node.scope) > 1:
                    factorized_input_sls = [
                        input_factory(Scope([sc]), num_input_units, num_channels)
                        for sc in node.scope
                    ]
                    input_sl = HadamardLayer(num_input_units, arity=len(factorized_input_sls))
                    layers.extend(factorized_input_sls)
                    in_layers[input_sl] = factorized_input_sls
                else:
                    input_sl = input_factory(node.scope, num_input_units, num_channels)
                num_units = num_sum_units if region_graph.region_outputs(node) else num_classes
                if sum_factory is None:
                    layers.append(input_sl)
                    node_to_layer[node] = input_sl
                    continue
                sum_sl = sum_factory(num_input_units, num_units)
                layers.append(input_sl)
                layers.append(sum_sl)
                in_layers[sum_sl] = [input_sl]
                node_to_layer[node] = sum_sl
            elif len(region_inputs) == 1:
                # Region node that is partitioned into one and only one way
                (ptn,) = region_inputs
                if sum_prod_builder_ is not None:
                    sum_prod_builder_(node, region_graph.partition_inputs(ptn))
                    continue
                num_units = num_sum_units if region_outputs else num_classes
                sum_input = node_to_layer[ptn]
                sum_sl = sum_factory(sum_input.num_output_units, num_units)
                layers.append(sum_sl)
                in_layers[sum_sl] = [sum_input]
                node_to_layer[node] = sum_sl
            else:  # len(node_inputs) > 1:
                # Region node with multiple partitionings
                num_units = num_sum_units if region_outputs else num_classes
                if sum_prod_builder_ is None:
                    sum_ins = [node_to_layer[ptn] for ptn in region_inputs]
                    mix_ins = [sum_factory(sli.num_output_units, num_units) for sli in sum_ins]
                    layers.extend(mix_ins)
                    for mix_sl, sli in zip(mix_ins, sum_ins):
                        in_layers[mix_sl] = [sli]
                else:
                    mix_ins = [
                        sum_prod_builder_(
                            node, region_graph.partition_inputs(cast(PartitionNode, ptn))
                        )
                        for ptn in region_inputs
                    ]
                mix_sl = SumLayer(
                    num_units,
                    num_units,
                    arity=len(mix_ins),
                    weight_factory=nary_sum_weight_factory,
                )
                layers.append(mix_sl)
                in_layers[mix_sl] = mix_ins
                node_to_layer[node] = mix_sl

        outputs = [node_to_layer[rgn] for rgn in region_graph.outputs]
        return cls(num_channels, layers, in_layers, outputs)

    @classmethod
    def from_hmm(
        cls,
        ordering: Sequence[int],
        input_factory: InputLayerFactory,
        weight_factory: ParameterFactory | None = None,
        num_channels: int = 1,
        num_units: int = 1,
        num_classes: int = 1,
    ) -> "Circuit":
        """Construct a symbolic circuit mimicking a hidden markov model (HMM) of
          a given variable ordering. Product Layers are of type
          [HadamardLayer][cirkit.symbolic.layers.HadamardLayer], and sum layers are of type
          [SumLayer][cirkit.symbolic.layers.SumLayer].

        Args:
            ordering: The input order of variables of the HMM.
            input_factory: A factory that builds input layers.
            weight_factory: The factory to construct the weight of sum layers. It can be None,
                or a parameter factory, i.e., a map from a shape to a symbolic parameter.
            num_channels: The number of channels for each variable.
            num_units: The number of sum units per sum layer.
            num_classes: The number of output classes.

        Returns:
            Circuit: A symbolic circuit.

        Raises:
            ValueError: order must consists of consistent numbers, starting from 0.
        """
        if max(ordering) != len(ordering) - 1 or min(ordering):
            raise ValueError("The 'ordering' of variables is not valid")

        layers: list[Layer] = []
        in_layers: dict[Layer, list[Layer]] = {}

        input_sl = input_factory(Scope([ordering[0]]), num_units, num_channels)
        layers.append(input_sl)
        sum_sl = SumLayer(num_units, num_units, weight_factory=weight_factory)
        layers.append(sum_sl)
        in_layers[sum_sl] = [input_sl]

        # Loop over the number of variables
        for i in range(1, len(ordering)):
            last_dense = layers[-1]

            input_sl = input_factory(Scope([ordering[i]]), num_units, num_channels)
            layers.append(input_sl)
            prod_sl = HadamardLayer(num_units, 2)
            layers.append(prod_sl)
            in_layers[prod_sl] = [last_dense, input_sl]

            num_units_out = num_units if i != len(ordering) - 1 else num_classes
            sum_sl = SumLayer(
                num_units,
                num_units_out,
                weight_factory=weight_factory,
            )
            layers.append(sum_sl)
            in_layers[sum_sl] = [prod_sl]

        return cls(num_channels, layers, in_layers, [layers[-1]])

    @classmethod
    def from_logic_circuit(
        cls,
        logic_graph: LogicGraph,
        *,
        literal_input_factory: InputLayerFactory = None,
        negated_literal_input_factory: InputLayerFactory = None,
        weight_factory: ParameterFactory | None = None,
        num_channels: int = 1,
    ) -> "Circuit":
        """
        Construct a symbolic circuit from a logic circuit graph.
        If input factories for literals and their negation are not provided the it
        falls back to a categorical input layer with two categories parametrized by
        the constant vector [0, 1] for a literal and [1, 0] for its negation.

        Args:
            logic_graph: The logic circuit graph.
            literal_input_factory: A factory that builds an input layer for literals.
            negated_literal_input_factory: A factory that builds an input layer for negated literals.
            weight_factory: The factory to construct the weight of sum layers. It can be None,
                or a parameter factory, i.e., a map from a shape to a symbolic parameter.
                If None is used, the default weight factory uses non-trainable unitary parameters,
                which instantiate a regular boolean logic graph.
            num_channels: The number of channels for each variable.

        Returns:
            Circuit: A symbolic circuit.

        Raises:
            ValueError: If only one of literal_input_factory and negated_literal_input_factory is specified.
        """
        in_layers: dict[Layer, Sequence[Layer]] = {}
        node_to_layer: dict[LogicCircuitNode, Layer] = {}

        if (literal_input_factory is None) ^ (negated_literal_input_factory is None):
            raise ValueError(
                "Either both 'literal_input_factory' and 'negated_literal_input_factory' \
                must be provided or none."
            )

        if literal_input_factory is None and negated_literal_input_factory is None:
            # default factory is locally imported when needed to avoid circular imports
            literal_input_factory = default_literal_input_factory(negated=False)
            negated_literal_input_factory = default_literal_input_factory(negated=True)

        if weight_factory is None:
            # default to unitary weights
            def weight_factory(n: tuple[int]) -> Parameter:
                # locally import numpy to avoid dependency on the whole file
                import numpy as np
                return Parameter.from_input(ConstantParameter(*n, value=np.ones(n)))

        # map each input literal to a symbolic input layer
        for i in logic_graph.inputs:
            match i:
                case LiteralNode():
                    node_to_layer[i] = literal_input_factory(
                        Scope([i.literal]), num_units=1, num_channels=num_channels
                    )
                case NegatedLiteralNode():
                    node_to_layer[i] = negated_literal_input_factory(
                        Scope([i.literal]), num_units=1, num_channels=num_channels
                    )

        for node in logic_graph.topological_ordering():
            match node:
                case ConjunctionNode():
                    product_node = HadamardLayer(1, arity=len(logic_graph.node_inputs(node)))

                    # if the product node contains Bottom Node as input then the
                    # the node can be pruned altogether since the result is trivial
                    if not any((isinstance(i, BottomNode) for i in logic_graph.node_inputs(node))):
                        # top nodes can be pruned from the product node since they do not contribute
                        in_layers[product_node] = [
                            node_to_layer[i]
                            for i in logic_graph.node_inputs(node)
                            if not isinstance(i, TopNode) and i in node_to_layer
                        ]
                        node_to_layer[node] = product_node

                case DisjunctionNode():
                    # bottom nodes can be pruned from the sum node since they do not contribute
                    sum_inputs = [
                        node_to_layer[i]
                        for i in logic_graph.node_inputs(node)
                        if not isinstance(i, BottomNode) and i in node_to_layer
                    ]

                    sum_node = SumLayer(1, 1, arity=len(sum_inputs), weight_factory=weight_factory)
                    in_layers[sum_node] = sum_inputs
                    node_to_layer[node] = sum_node

        # since we are pruning nodes during the mapping procedure, it might happen that some layers
        # are not connected to any other layer, and they are not the output node
        # in that case, they can be safely removed
        in_layers = {
            layer: layer_inputs
            for layer, layer_inputs in in_layers.items()
            if any((layer in l for l in in_layers.values()))
            or layer == node_to_layer[logic_graph.output]
        }

        layers = list(set(itertools.chain(*in_layers.values())).union(in_layers.keys()))

        return cls(num_channels, layers, in_layers, [node_to_layer[logic_graph.output]])


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
