import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from cirkit.symbolic.layers import (
    DenseLayer,
    HadamardLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import ParameterFactory
from cirkit.templates.region_graph import PartitionNode, RegionGraph, RegionNode, RGNode
from cirkit.utils.algorithms import (
    BiRootedDiAcyclicGraph,
    DiAcyclicGraph,
    bfs,
    topological_ordering,
)
from cirkit.utils.orderedset import OrderedSet
from cirkit.utils.scope import Scope

AbstractCircuitOperator = IntEnum  # TODO: switch to StrEnum (>=py3.11) or better alternative


class CircuitOperator(AbstractCircuitOperator):
    """Types of Symolic operations on circuits."""

    def _generate_next_value_(self, start: int, count: int, last_values: list) -> int:
        return -(
            count + 1
        )  # Enumerate negative integers as the user can extend them with non-negative ones

    MERGE = auto()
    INTEGRATION = auto()
    DIFFERENTIATION = auto()
    MULTIPLICATION = auto()
    CONJUGATION = auto()


@dataclass(frozen=True)
class CircuitOperation:
    """The Symbolic operation applied on a SymCircuit."""

    operator: AbstractCircuitOperator
    operands: Tuple["Circuit", ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBlock(BiRootedDiAcyclicGraph[Layer]):
    def __init__(
        self,
        layers: List[Layer],
        in_layers: Dict[Layer, List[Layer]],
        outputs: List[Layer],
        *,
        topologically_ordered: bool = False,
    ):
        super().__init__(layers, in_layers, outputs, topologically_ordered=topologically_ordered)

    def layer_inputs(self, l: Layer) -> List[Layer]:
        return self.node_inputs(l)

    def layer_outputs(self, l: Layer) -> List[Layer]:
        return self.node_outputs(l)

    @property
    def layers_inputs(self) -> Dict[Layer, List[Layer]]:
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Dict[Layer, List[Layer]]:
        return self.nodes_outputs

    @property
    def layers(self) -> List[Layer]:
        return self.nodes

    @property
    def inner_layers(self) -> Iterator[Union[SumLayer, ProductLayer]]:
        """Inner (non-input) layers in the circuit."""
        return (l for l in self.layers if isinstance(l, (SumLayer, ProductLayer)))

    @property
    def sum_layers(self) -> Iterator[SumLayer]:
        """Sum layers in the circuit, which are always inner layers."""
        return (l for l in self.layers if isinstance(l, SumLayer))

    @property
    def product_layers(self) -> Iterator[ProductLayer]:
        """Product layers in the circuit, which are always inner layers."""
        return (l for l in self.layers if isinstance(l, ProductLayer))

    @staticmethod
    def from_layer(sl: Layer) -> "CircuitBlock":
        return CircuitBlock([sl], {}, [sl], topologically_ordered=True)

    @staticmethod
    def from_layer_composition(*sl: Layer) -> "CircuitBlock":
        layers = list(sl)
        in_layers = {}
        assert len(layers) > 1, "Expected a composition of at least 2 layers"
        for i, l in enumerate(layers):
            in_layers[l] = [layers[i - 1]] if i - 1 >= 0 else []
        return CircuitBlock(layers, in_layers, [sl[-1]], topologically_ordered=True)


class InputLayerFactory(Protocol):
    def __call__(self, scope: Scope, num_units: int, num_channels: int) -> InputLayer:
        ...


class SumLayerFactory(Protocol):
    def __call__(self, scope: Scope, num_input_units: int, num_output_units: int) -> SumLayer:
        ...


class ProductLayerFactory(Protocol):
    def __call__(self, scope: Scope, num_input_units: int, arity: int) -> ProductLayer:
        ...


class MixingLayerFactory(Protocol):
    def __call__(self, scope: Scope, num_units: int, arity: int) -> SumLayer:
        ...


class Circuit(DiAcyclicGraph[Layer]):
    """The symbolic representation of a circuit."""

    def __init__(
        self,
        scope: Scope,
        num_channels: int,
        layers: List[Layer],
        in_layers: Dict[Layer, List[Layer]],
        outputs: List[Layer],
        *,
        operation: Optional[CircuitOperation] = None,
        topologically_ordered: bool = False,
    ) -> None:
        super().__init__(layers, in_layers, outputs, topologically_ordered=topologically_ordered)
        self.scope = scope
        self.num_channels = num_channels
        self.operation = operation

    @property
    def num_variables(self) -> int:
        return len(self.scope)

    def layer_inputs(self, l: Layer) -> List[Layer]:
        return self.node_inputs(l)

    def layer_outputs(self, l: Layer) -> List[Layer]:
        return self.node_outputs(l)

    @property
    def layers_inputs(self) -> Dict[Layer, List[Layer]]:
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Dict[Layer, List[Layer]]:
        return self.nodes_outputs

    @property
    def layers(self) -> List[Layer]:
        return self.nodes

    #######################################    Layer views    ######################################
    # These are iterable views of the nodes in the SymC, and the topological order is guaranteed
    # (by a stronger ordering). For efficiency, all these views are iterators (implemented as a
    # container iter or a generator), so that they can be chained for iteration without
    # instantiating intermediate containers.

    @property
    def inner_layers(self) -> Iterator[Union[SumLayer, ProductLayer]]:
        """Inner (non-input) layers in the circuit."""
        return (l for l in self.layers if isinstance(l, (SumLayer, ProductLayer)))

    @property
    def sum_layers(self) -> Iterator[SumLayer]:
        """Sum layers in the circuit, which are always inner layers."""
        return (l for l in self.layers if isinstance(l, SumLayer))

    @property
    def product_layers(self) -> Iterator[ProductLayer]:
        """Product layers in the circuit, which are always inner layers."""
        return (l for l in self.layers if isinstance(l, ProductLayer))

    ##################################### Structural properties ####################################

    @cached_property
    def is_smooth(self) -> bool:
        return all(
            sum_sl.scope == in_sl.scope
            for sum_sl in self.sum_layers
            for in_sl in self.layer_inputs(sum_sl)
        )

    @cached_property
    def is_decomposable(self) -> bool:
        return not any(
            lhs_in_sl.scope & rhs_in_sl.scope
            for prod_sl in self.product_layers
            for lhs_in_sl, rhs_in_sl in itertools.combinations(self.layer_inputs(prod_sl), 2)
        )

    @cached_property
    def is_structured_decomposable(self) -> bool:
        # Structured-decomposability is self-compatiblity
        return self.is_compatible(self)

    @cached_property
    def is_omni_compatible(self) -> bool:
        if not self.is_smooth:
            return False
        if not self.is_decomposable:
            return False
        # TODO
        return False

    def is_compatible(self, oth: "Circuit", scope: Optional[Iterable[int]] = None) -> bool:
        if not self.is_smooth:
            return False
        if not self.is_decomposable:
            return False
        # TODO
        return True

    @classmethod
    def from_operation(
        cls,
        scope: Scope,
        num_channels: int,
        blocks: List[CircuitBlock],
        in_blocks: Dict[CircuitBlock, List[CircuitBlock]],
        output_blocks: List[CircuitBlock],
        operation: CircuitOperation,
        *,
        topologically_ordered: bool = False,
    ):
        # Unwrap blocks into layers (as well as their connections)
        layers = [l for b in blocks for l in b.layers]
        in_layers = defaultdict(list)
        outputs = [b.output for b in output_blocks]

        # Retrieve connections between layers from connections between circuit blocks
        for b in blocks:
            in_layers[b.input].extend(bi.output for bi in in_blocks.get(b, []))
            for l in b.layers:
                in_layers[l].extend(b.layer_inputs(l))
        # Build the circuit and set the operation
        return cls(
            scope,
            num_channels,
            layers,
            in_layers,
            outputs,
            operation=operation,
            topologically_ordered=topologically_ordered
            and all(b.is_topologically_ordered for b in blocks),
        )

    @classmethod
    def from_region_graph(
        cls,
        region_graph: RegionGraph,
        *,
        input_factory: InputLayerFactory,
        sum_product: Optional[str] = None,
        dense_weight_factory: Optional[ParameterFactory] = None,
        sum_factory: Optional[SumLayerFactory] = None,
        prod_factory: Optional[ProductLayerFactory] = None,
        mixing_factory: Optional[MixingLayerFactory] = None,
        num_channels: int = 1,
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
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
            dense_weight_factory: The factory to construct the weight of the sum-product layer abstraction.
                It can be None, or a parameter factory, i.e., a map from a shape to a symbolic parameter.
            sum_factory: A factory that builds a sum layer. It can be None.
            prod_factory: A factory that builds a product layer. It can be None.
            mixing_factory: A factory that builds a mixing layer. It can be None if the given region graph
                does not have any region node being decomposed into more than one partitioning.
            num_channels: The number of channels for each variable.
            num_input_units: The number of input units.
            num_sum_units: The number of sum units per sum layer.
            num_classes: The number of output classes.

        Returns:
            Circuit: A symbolic circuit.

        Raises:
            ValueError: If both 'sum_product' and layer factories are specified, or none of them.
            ValueError: If the mixing factory is required, but it was not given.
        """
        if (sum_factory is None and prod_factory is not None) or (
            sum_factory is not None and prod_factory is None
        ):
            raise ValueError(
                "Both 'sum_factory' and 'prod_factory' must be specified or none of them"
            )
        if sum_product is None and (sum_factory is None or prod_factory is None):
            raise ValueError(
                "If 'sum_product' is not given, then both 'sum_factory' and 'prod_factory' must be specified"
            )
        if sum_product is not None and (sum_factory is not None or prod_factory is not None):
            raise ValueError(
                "At most one between 'sum_product' and the pair 'sum_factory' and 'prod_factory' must be specified"
            )

        layers: List[Layer] = []
        in_layers: Dict[Layer, List[Layer]] = {}
        rgn_to_layers: Dict[RGNode, Layer] = {}

        def build_cp_(
            rgn: RegionNode, rgn_partitioning: OrderedSet[RegionNode], num_output_units: int
        ) -> HadamardLayer:
            layer_ins = [rgn_to_layers[rgn_in] for rgn_in in rgn_partitioning]
            denses = [
                DenseLayer(
                    rgn_in.scope,
                    rgn_to_layers[rgn_in].num_output_units,
                    num_output_units,
                    weight_factory=dense_weight_factory,
                )
                for rgn_in in rgn_partitioning
            ]
            hadamard = HadamardLayer(rgn.scope, num_output_units, arity=len(rgn_partitioning))
            layers.extend(denses)
            layers.append(hadamard)
            in_layers[hadamard] = denses
            for d, li in zip(denses, layer_ins):
                in_layers[d] = [li]
            rgn_to_layers[rgn] = hadamard
            return hadamard

        def build_cp_transposed_(
            rgn: RegionNode, rgn_partitioning: OrderedSet[RegionNode], num_output_units: int
        ) -> DenseLayer:
            layer_ins = [rgn_to_layers[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list(set([li.num_output_units for li in layer_ins]))
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a CP transposed layer, as the inputs would have different units"
                )
            hadamard = HadamardLayer(rgn.scope, num_in_units[0], arity=len(rgn_partitioning))
            dense = DenseLayer(
                rgn.scope, num_in_units[0], num_output_units, weight_factory=dense_weight_factory
            )
            layers.append(hadamard)
            layers.append(dense)
            in_layers[hadamard] = layer_ins
            in_layers[dense] = [hadamard]
            rgn_to_layers[rgn] = dense
            return dense

        def build_tucker_(
            rgn: RegionNode, rgn_partitioning: OrderedSet[RegionNode], num_output_units: int
        ) -> DenseLayer:
            layer_ins = [rgn_to_layers[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list(set([li.num_output_units for li in layer_ins]))
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a Tucker layer, as the inputs would have different units"
                )
            kronecker = KroneckerLayer(rgn.scope, num_in_units[0], arity=len(rgn_partitioning))
            dense = DenseLayer(
                rgn.scope, num_in_units[0], num_output_units, weight_factory=dense_weight_factory
            )
            layers.append(kronecker)
            layers.append(dense)
            in_layers[kronecker] = layer_ins
            in_layers[dense] = [kronecker]
            rgn_to_layers[rgn] = dense
            return dense

        sum_prod_builder_: Optional[Callable[[Scope, OrderedSet[RegionNode], int], Layer]]
        if sum_product is None:
            sum_prod_builder_ = None
        elif sum_product == "cp":
            sum_prod_builder_ = build_cp_
        elif sum_product == "cp-t":
            sum_prod_builder_ = build_cp_transposed_
        elif sum_product == "tucker":
            sum_prod_builder_ = build_tucker_
        else:
            raise ValueError(f"Unknown sum-product layer abstraction called {sum_product}")

        # Loop through the region graph nodes, which are already sorted in a topological ordering
        for rgn in region_graph.nodes:
            if isinstance(rgn, RegionNode) and not rgn.inputs:  # Input region node
                input_sl = input_factory(rgn.scope, num_input_units, num_channels)
                num_output_units = num_sum_units if rgn.outputs else num_classes
                if sum_factory is None:
                    layers.append(input_sl)
                    rgn_to_layers[rgn] = input_sl
                    continue
                sum_sl = sum_factory(rgn.scope, num_input_units, num_output_units)
                layers.append(input_sl)
                layers.append(sum_sl)
                in_layers[sum_sl] = [input_sl]
                rgn_to_layers[rgn] = sum_sl
            elif isinstance(rgn, PartitionNode):  # Partition node
                # If a sum-product layer abstraction has been specified, then just skip partition nodes
                if sum_prod_builder_ is not None:
                    continue
                assert prod_factory is not None
                prod_inputs = [rgn_to_layers[rgn_in] for rgn_in in rgn.inputs]
                prod_sl = prod_factory(rgn.scope, num_sum_units, len(prod_inputs))
                layers.append(prod_sl)
                in_layers[prod_sl] = prod_inputs
                rgn_to_layers[rgn] = prod_sl
            elif isinstance(rgn, RegionNode) and len(rgn.inputs) == 1:  # Region node
                num_units = num_sum_units if rgn.outputs else num_classes
                (rgn_in,) = rgn.inputs
                if sum_prod_builder_ is not None:
                    sum_prod_builder_(rgn, rgn_in.inputs, num_units)
                    continue
                sum_input = rgn_to_layers[rgn_in]
                sum_sl = sum_factory(rgn.scope, sum_input.num_output_units, num_units)
                layers.append(sum_sl)
                in_layers[sum_sl] = [sum_input]
                rgn_to_layers[rgn] = sum_sl
            elif (
                isinstance(rgn, RegionNode) and len(rgn.inputs) > 1
            ):  # Region with multiple partitionings
                num_units = num_sum_units if rgn.outputs else num_classes
                if mixing_factory is None:
                    raise ValueError(
                        "A mixing layer factory must be specified to overparameterize multiple region partitionings"
                    )
                if sum_prod_builder_ is not None:
                    mix_ins = [
                        sum_prod_builder_(rgn, rgn_in.inputs, num_units) for rgn_in in rgn.inputs
                    ]
                else:
                    sum_ins = [rgn_to_layers[rgn_in] for rgn_in in rgn.inputs]
                    mix_ins = [
                        sum_factory(rgn.scope, sli.num_output_units, num_units) for sli in sum_ins
                    ]
                    layers.extend(mix_ins)
                    for mix_sl, sli in zip(mix_ins, sum_ins):
                        in_layers[mix_sl] = [sli]
                mix_sl = mixing_factory(rgn.scope, num_units, len(mix_ins))
                layers.append(mix_sl)
                in_layers[mix_sl] = mix_ins
                rgn_to_layers[rgn] = mix_sl
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "Region graph nodes must be either region or partition nodes"

        outputs = [rgn_to_layers[rgn] for rgn in region_graph.output_nodes]
        return cls(
            region_graph.scope,
            num_channels,
            layers,
            in_layers,
            outputs,
            topologically_ordered=True,
        )


def pipeline_topological_ordering(roots: Sequence[Circuit]) -> Iterator[Circuit]:
    def operands_fn(sc: Circuit) -> Tuple[Circuit, ...]:
        return () if sc.operation is None else sc.operation.operands

    return topological_ordering(bfs(roots, incomings_fn=operands_fn), incomings_fn=operands_fn)


class StructuralPropertyError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)
