import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from cirkit.symbolic.initializers import ConstantInitializer
from cirkit.symbolic.layers import (
    DenseLayer,
    HadamardLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    MixingLayer,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import Parameter, ParameterFactory, TensorParameter
from cirkit.templates.region_graph import PartitionNode, RegionGraph, RegionGraphNode, RegionNode
from cirkit.utils.algorithms import DiAcyclicGraph, RootedDiAcyclicGraph, bfs, topological_ordering
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


class CircuitOperator(IntEnum):
    """The available symbolic operators defined over circuits."""

    CONCATENATE = auto()
    """The concatenation operator defined over many circuits."""
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
    operands: Tuple["Circuit", ...]
    """The circuit operands of the operation."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata of the operation."""


class CircuitBlock(RootedDiAcyclicGraph[Layer]):
    """The circuit block data structure. A circuit block is a fragment of a symbolic circuit,
    consisting of a single root (or output) layer. A circuit block can be of only two types:
    (1) a circuit block whose layers that do not have any inputs must all be input layers, and
    (2) a circuit block where there is one and only one layer that does not have any other input,
    which can be either a sum or product layer.
    """

    def __init__(self, layers: List[Layer], in_layers: Dict[Layer, List[Layer]], output: Layer):
        """Initializes a circuit block.

        Args:
            layers: The list of layers in the block.
            in_layers: A dictionary containing the list of inputs to each layer.
            output: The root (or output) of the circuit block.
        """
        super().__init__(layers, in_layers, [output])

    def layer_inputs(self, l: Layer) -> List[Layer]:
        """Retrieves the inputs to a layer.

        Args:
            l: The layer.

        Returns:
            List[Layer]: The list of inputs.
        """
        return self.node_inputs(l)

    def layer_outputs(self, l: Layer) -> List[Layer]:
        """Retrieves the outputs of a layer.

        Args:
            l: The layer.

        Returns:
            List[Layer]: The list of outputs.
        """
        return self.node_outputs(l)

    @property
    def layers_inputs(self) -> Dict[Layer, List[Layer]]:
        """Retrieves the dictionary containing the list of inputs to each layer.

        Returns:
            Dict[Layer, List[Layer]]:
        """
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Dict[Layer, List[Layer]]:
        """Retrieves the dictionary containing the list of outputs of each layer.

        Returns:
            Dict[Layer, List[Layer]]:
        """
        return self.nodes_outputs

    @property
    def layers(self) -> List[Layer]:
        """Retrieves the list of layers.

        Returns:
            List[layer]:
        """
        return self.nodes

    @property
    def inner_layers(self) -> Iterator[Union[SumLayer, ProductLayer]]:
        """Retrieves an iterator over inner layers (i.e., layers that have at least one input).

        Returns:
            Iterator[Union[SumLayer, ProductLayer]]:
        """
        return (l for l in self.layers if isinstance(l, (SumLayer, ProductLayer)))

    @property
    def sum_layers(self) -> Iterator[SumLayer]:
        """Retrieves an iterator over sum layers.

        Returns:
            Iterator[SumLayer]:
        """
        return (l for l in self.layers if isinstance(l, SumLayer))

    @property
    def product_layers(self) -> Iterator[ProductLayer]:
        """Retrieves an iterator over product layers.

        Returns:
            Iterator[ProductLayer]:
        """
        return (l for l in self.layers if isinstance(l, ProductLayer))

    @staticmethod
    def from_layer(l: Layer) -> "CircuitBlock":
        """Instantiate a circuit block from a single layer.

        Args:
            l: The layer.

        Returns:
            CircuitBlock: The circuit block consisting of only one layer.
        """
        return CircuitBlock([l], {}, l)

    @staticmethod
    def from_layer_composition(*ls: Layer) -> "CircuitBlock":
        """Instantiate a circuit block from a composition of multiple layers.
         The ordering of the composition is given by the ordering of the layers.

        Args:
            ls: A sequence of layers.

        Returns:
            CircuitBlock: The circuit block consisting of a composition of layers.
        """
        layers = list(ls)
        in_layers = {}
        assert len(layers) > 1, "Expected a composition of at least 2 layers"
        for i, l in enumerate(layers):
            in_layers[l] = [layers[i - 1]] if i - 1 >= 0 else []
        return CircuitBlock(layers, in_layers, layers[-1])


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

    def __call__(self, scope: Scope, num_input_units: int, num_output_units: int) -> SumLayer:
        """Constructs a sum layer.

        Args:
            scope: The scope of the layer.
            num_input_units: The number of units in each layer that is an input.
            num_output_units: The number of sum units in the layer.

        Returns:
            SumLayer: A sum layer.
        """


class ProductLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs product layers."""

    def __call__(self, scope: Scope, num_input_units: int, arity: int) -> ProductLayer:
        """Constructs a product layer.

        Args:
            scope: The scope of the layer.
            num_input_units: The number of units in each layer that is an input.
            arity: The number of input layers.

        Returns:
            ProductLayer: A product layer.
        """


class MixingLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs mixing layers,
    i.e., layers computing a linear sum over two or more input layers, and
    that have the same number of sum units as the units in each input layer."""

    def __call__(self, scope: Scope, num_units: int, arity: int) -> SumLayer:
        """Constructs a mixing layer.

        Args:
            scope: The scope of the layer.
            num_units: The number of units in each layer that is an input.
            arity: The number of input layers.

        Returns:
            SumLayer: A mixing layer.
        """


class Circuit(DiAcyclicGraph[Layer]):
    """The symbolic circuit representation."""

    def __init__(
        self,
        scope: Scope,
        num_channels: int,
        layers: List[Layer],
        in_layers: Dict[Layer, List[Layer]],
        outputs: List[Layer],
        *,
        operation: Optional[CircuitOperation] = None,
    ) -> None:
        """Initializes a symbolic circuit.

        Args:
            scope: The variables scope of the circuit.
            num_channels: The number of channels for each variable.
            layers: The list of symbolic layers.
            in_layers: A dictionary containing the list of inputs to each layer.
            outputs: The output layers of the circuit.
            operation: The optional operation the circuit has been obtained through.
        """
        super().__init__(layers, in_layers, outputs)
        self.scope = scope
        self.num_channels = num_channels
        self.operation = operation

    @property
    def num_variables(self) -> int:
        """Retrieves the number of variables the circuit is defined on.

        Returns:
            int:
        """
        return len(self.scope) * self.num_channels

    def layer_inputs(self, l: Layer) -> List[Layer]:
        """Retrieves the inputs to a layer.

        Args:
            l: The layer.

        Returns:
            List[Layer]: The list of inputs.
        """
        return self.node_inputs(l)

    def layer_outputs(self, l: Layer) -> List[Layer]:
        """Retrieves the outputs of a layer.

        Args:
            l: The layer.

        Returns:
            List[Layer]: The list of outputs.
        """
        return self.node_outputs(l)

    @property
    def layers_inputs(self) -> Dict[Layer, List[Layer]]:
        """Retrieves the dictionary containing the list of inputs to each layer.

        Returns:
            Dict[Layer, List[Layer]]:
        """
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Dict[Layer, List[Layer]]:
        """Retrieves the dictionary containing the list of outputs of each layer.

        Returns:
            Dict[Layer, List[Layer]]:
        """
        return self.nodes_outputs

    @property
    def layers(self) -> List[Layer]:
        """Retrieves the list of layers.

        Returns:
            List[layer]:
        """
        return self.nodes

    @property
    def inner_layers(self) -> Iterator[Union[SumLayer, ProductLayer]]:
        """Retrieves an iterator over inner layers (i.e., non-input layers).

        Returns:
            Iterator[Union[SumLayer, ProductLayer]]:
        """
        return (l for l in self.layers if isinstance(l, (SumLayer, ProductLayer)))

    @property
    def sum_layers(self) -> Iterator[SumLayer]:
        """Retrieves an iterator over sum layers.

        Returns:
            Iterator[SumLayer]:
        """
        return (l for l in self.layers if isinstance(l, SumLayer))

    @property
    def product_layers(self) -> Iterator[ProductLayer]:
        """Retrieves an iterator over product layers.

        Returns:
            Iterator[ProductLayer]:
        """
        return (l for l in self.layers if isinstance(l, ProductLayer))

    ##################################### Structural properties ####################################

    @cached_property
    def is_smooth(self) -> bool:
        """Check if the circuit is smooth.

        Returns:
            bool: True if the circuit is smooth and False otherwise.
        """
        return all(
            sum_sl.scope == in_sl.scope
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
            lhs_in_sl.scope & rhs_in_sl.scope
            for prod_sl in self.product_layers
            for lhs_in_sl, rhs_in_sl in itertools.combinations(self.layer_inputs(prod_sl), 2)
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

    @classmethod
    def from_operation(
        cls,
        scope: Scope,
        num_channels: int,
        blocks: List[CircuitBlock],
        in_blocks: Dict[CircuitBlock, List[CircuitBlock]],
        output_blocks: List[CircuitBlock],
        *,
        operation: CircuitOperation,
    ) -> "Circuit":
        """Constructs a circuit that resulted from an operation over other circuits.

        Args:
            scope: The variables scope the circuit is defined on.
            num_channels: The number of channels per variable.
            blocks: The list of circuit blocks.
            in_blocks: A dictionary containing the list of block inputs to each circuit block.
            output_blocks: The outputs blocks of the circuit.
            operation: A circuit operation containing the information of the operation.

        Returns:
            Circuit: A symbolic circuit.

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
            for l in b.layers:
                in_layers[l].extend(b.layer_inputs(l))
        # Build the circuit and set the operation
        return cls(scope, num_channels, layers, in_layers, outputs, operation=operation)

    @classmethod
    def from_region_graph(
        cls,
        region_graph: RegionGraph,
        *,
        input_factory: InputLayerFactory,
        sum_product: Optional[str] = None,
        sum_weight_factory: Optional[ParameterFactory] = None,
        sum_factory: Optional[SumLayerFactory] = None,
        prod_factory: Optional[ProductLayerFactory] = None,
        mixing_factory: Optional[MixingLayerFactory] = None,
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
            sum_weight_factory: The factory to construct the weight of the sum-product layer
                abstraction and mixing layers. It can be None, or a parameter factory, i.e., a map
                from a shape to a symbolic parameter.
            sum_factory: A factory that builds a sum layer. It can be None.
            prod_factory: A factory that builds a product layer. It can be None.
            mixing_factory: A factory that builds a mixing layer, i.e., a layer used to parameterize
                a region node that is decomposed into more than one partitioning. If it is None,
                then it is assumed to be a factory that builds a
                [MixingLayer][cirkit.symbolic.layers.MixingLayer].
                If 'sum_weight_factory' is None then the weight parameters are not
                learnable and are initialized to the constant 1/H, where H is the arity of the
                mixing layer, i.e., the number of input layers. Otherwise, 'sum_weight_factory' is
                used to construct the weights of the mixing layers.
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

        layers: List[Layer] = []
        in_layers: Dict[Layer, List[Layer]] = {}
        node_to_layer: Dict[RegionGraphNode, Layer] = {}

        def default_mixing_layer_factory(scope: Scope, num_units: int, arity: int) -> MixingLayer:
            if sum_weight_factory is None:
                weight = Parameter.from_leaf(
                    TensorParameter(
                        num_units,
                        arity,
                        initializer=ConstantInitializer(1.0 / arity),
                        learnable=False,
                    )
                )
                return MixingLayer(scope, num_units, arity, weight=weight)
            return MixingLayer(scope, num_units, arity, weight_factory=sum_weight_factory)

        def build_cp_(
            rgn: RegionNode, rgn_partitioning: List[RegionNode], num_output_units: int
        ) -> HadamardLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            denses = [
                DenseLayer(
                    rgn_in.scope,
                    node_to_layer[rgn_in].num_output_units,
                    num_output_units,
                    weight_factory=sum_weight_factory,
                )
                for rgn_in in rgn_partitioning
            ]
            hadamard = HadamardLayer(rgn.scope, num_output_units, arity=len(rgn_partitioning))
            layers.extend(denses)
            layers.append(hadamard)
            in_layers[hadamard] = denses
            for d, li in zip(denses, layer_ins):
                in_layers[d] = [li]
            node_to_layer[rgn] = hadamard
            return hadamard

        def build_cp_transposed_(
            rgn: RegionNode, rgn_partitioning: List[RegionNode], num_output_units: int
        ) -> DenseLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list(set(li.num_output_units for li in layer_ins))
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a CP transposed layer, as the inputs would have different units"
                )
            hadamard = HadamardLayer(rgn.scope, num_in_units[0], arity=len(rgn_partitioning))
            dense = DenseLayer(
                rgn.scope, num_in_units[0], num_output_units, weight_factory=sum_weight_factory
            )
            layers.append(hadamard)
            layers.append(dense)
            in_layers[hadamard] = layer_ins
            in_layers[dense] = [hadamard]
            node_to_layer[rgn] = dense
            return dense

        def build_tucker_(
            rgn: RegionNode, rgn_partitioning: List[RegionNode], num_output_units: int
        ) -> DenseLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list(set(li.num_output_units for li in layer_ins))
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a Tucker layer, as the inputs would have different units"
                )
            kronecker = KroneckerLayer(rgn.scope, num_in_units[0], arity=len(rgn_partitioning))
            dense = DenseLayer(
                rgn.scope, num_in_units[0], num_output_units, weight_factory=sum_weight_factory
            )
            layers.append(kronecker)
            layers.append(dense)
            in_layers[kronecker] = layer_ins
            in_layers[dense] = [kronecker]
            node_to_layer[rgn] = dense
            return dense

        # Set the mixing factory as the default one (see above), if not given
        if mixing_factory is None:
            mixing_factory = default_mixing_layer_factory

        # Set the sum-product layer builder, if necessary
        sum_prod_builder_: Optional[Callable[[RegionNode, List[RegionNode], int], Layer]]
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
            node_inputs = region_graph.node_inputs(node)
            node_outputs = region_graph.node_outputs(node)
            if isinstance(node, RegionNode) and not node_inputs:  # Input region node
                if factorize_multivariate and len(node.scope) > 1:
                    factorized_input_sls = [
                        input_factory(Scope([sc]), num_input_units, num_channels)
                        for sc in node.scope
                    ]
                    input_sl = HadamardLayer(
                        node.scope, num_input_units, arity=len(factorized_input_sls)
                    )
                    layers.extend(factorized_input_sls)
                    in_layers[input_sl] = factorized_input_sls
                else:
                    input_sl = input_factory(node.scope, num_input_units, num_channels)
                num_output_units = num_sum_units if node_outputs else num_classes
                if sum_factory is None:
                    layers.append(input_sl)
                    node_to_layer[node] = input_sl
                    continue
                sum_sl = sum_factory(node.scope, num_input_units, num_output_units)
                layers.append(input_sl)
                layers.append(sum_sl)
                in_layers[sum_sl] = [input_sl]
                node_to_layer[node] = sum_sl
            elif isinstance(node, PartitionNode):  # Partition node
                # If a sum-product layer abstraction has been specified,
                # then just skip partition nodes
                if sum_prod_builder_ is not None:
                    continue
                assert prod_factory is not None
                prod_inputs = [node_to_layer[rgn] for rgn in node_inputs]
                prod_sl = prod_factory(node.scope, num_sum_units, len(prod_inputs))
                layers.append(prod_sl)
                in_layers[prod_sl] = prod_inputs
                node_to_layer[node] = prod_sl
            elif isinstance(node, RegionNode) and len(node_inputs) == 1:  # Region node
                num_units = num_sum_units if node_outputs else num_classes
                (ptn,) = node_inputs
                if sum_prod_builder_ is not None:
                    sum_prod_builder_(node, region_graph.partition_inputs(ptn), num_units)
                    continue
                sum_input = node_to_layer[ptn]
                sum_sl = sum_factory(node.scope, sum_input.num_output_units, num_units)
                layers.append(sum_sl)
                in_layers[sum_sl] = [sum_input]
                node_to_layer[node] = sum_sl
            elif (
                isinstance(node, RegionNode) and len(node_inputs) > 1
            ):  # Region with multiple partitionings
                num_units = num_sum_units if node_outputs else num_classes
                if sum_prod_builder_ is None:
                    sum_ins = [node_to_layer[ptn] for ptn in node_inputs]
                    mix_ins = [
                        sum_factory(node.scope, sli.num_output_units, num_units) for sli in sum_ins
                    ]
                    layers.extend(mix_ins)
                    for mix_sl, sli in zip(mix_ins, sum_ins):
                        in_layers[mix_sl] = [sli]
                else:
                    mix_ins = [
                        sum_prod_builder_(
                            node, region_graph.partition_inputs(cast(PartitionNode, ptn)), num_units
                        )
                        for ptn in node_inputs
                    ]
                mix_sl = mixing_factory(node.scope, num_units, len(mix_ins))
                layers.append(mix_sl)
                in_layers[mix_sl] = mix_ins
                node_to_layer[node] = mix_sl
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                raise ValueError("Region graph nodes must be either region or partition nodes")

        outputs = [node_to_layer[rgn] for rgn in region_graph.outputs]
        return cls(region_graph.scope, num_channels, layers, in_layers, outputs)

    @classmethod
    def from_hmm(
        cls,
        ordering: Sequence[int],
        input_factory: InputLayerFactory,
        weight_factory: Optional[ParameterFactory] = None,
        num_channels: int = 1,
        num_units: int = 1,
        num_classes: int = 1,
    ) -> "Circuit":
        """Construct a symbolic circuit mimicking a hidden markov model (HMM) of
          a given variable ordering. Product Layers are of type
          [HadamardLayer][cirkit.symbolic.layers.HadamardLayer], and sum layers are of type
          [DenseLayer][cirkit.symbolic.layers.DenseLayer].

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

        layers: List[Layer] = []
        in_layers: Dict[Layer, List[Layer]] = {}

        input_sl = input_factory(Scope([ordering[0]]), num_units, num_channels)
        layers.append(input_sl)
        sum_sl = DenseLayer(
            Scope([ordering[0]]), num_units, num_units, weight_factory=weight_factory
        )
        layers.append(sum_sl)
        in_layers[sum_sl] = [input_sl]

        # Loop over the number of variables
        for i in range(1, len(ordering)):
            last_dense = layers[-1]

            input_sl = input_factory(Scope([ordering[i]]), num_units, num_channels)
            layers.append(input_sl)
            prod_sl = HadamardLayer(Scope(ordering[: (i + 1)]), num_units, 2)
            layers.append(prod_sl)
            in_layers[prod_sl] = [last_dense, input_sl]

            num_units_out = num_units if i != len(ordering) - 1 else num_classes
            sum_sl = DenseLayer(
                Scope(ordering[: (i + 1)]),
                num_units,
                num_units_out,
                weight_factory=weight_factory,
            )
            layers.append(sum_sl)
            in_layers[sum_sl] = [prod_sl]

        return cls(Scope(ordering), num_channels, layers, in_layers, [layers[-1]])


def is_compatible(sc1: Circuit, sc2: Circuit) -> bool:
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

    def operands_fn(sc: Circuit) -> Tuple[Circuit, ...]:
        return () if sc.operation is None else sc.operation.operands

    return topological_ordering(bfs(roots, incomings_fn=operands_fn), incomings_fn=operands_fn)


def _scope_factorizations(sc: Circuit) -> Dict[Scope, Set[Tuple[Scope, ...]]]:
    # For each product layer, retrieves how it factorizes its scope
    scope_factorizations: Dict[Scope, Set[Tuple[Scope, ...]]] = defaultdict(set)
    for sl in sc.product_layers:
        scope_factorizations[sl.scope].add(tuple(sorted(sli.scope for sli in sc.layer_inputs(sl))))
    return scope_factorizations


def _are_compatible(
    sfs1: Dict[Scope, Set[Tuple[Scope, ...]]], sfs2: Dict[Scope, Set[Tuple[Scope, ...]]]
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
