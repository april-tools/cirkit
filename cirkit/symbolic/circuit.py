import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, Union

from cirkit.symbolic.layers import InputLayer, Layer, ProductLayer, SumLayer
from cirkit.templates.region_graph import PartitionNode, RegionGraph, RegionNode, RGNode
from cirkit.utils.algorithms import DiAcyclicGraph, RootedDiAcyclicGraph, bfs, topological_ordering
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


class CircuitBlock(RootedDiAcyclicGraph[Layer]):
    def __init__(
        self,
        layers: List[Layer],
        in_layers: Dict[Layer, List[Layer]],
        output: Layer,
        *,
        topologically_ordered: bool = False,
    ):
        super().__init__(layers, in_layers, [output], topologically_ordered=topologically_ordered)

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
        return CircuitBlock([sl], {}, sl, topologically_ordered=True)

    @staticmethod
    def from_layer_composition(*sl: Layer) -> "CircuitBlock":
        layers = list(sl)
        in_layers = {}
        assert len(layers) > 1, "Expected a composition of at least 2 layers"
        for i, l in enumerate(layers):
            in_layers[l] = [layers[i - 1]] if i - 1 >= 0 else []
        return CircuitBlock(layers, in_layers, sl[-1], topologically_ordered=True)


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
            b_layer_inputs = list(b.inputs)
            block_ins = in_blocks.get(b, [])
            if len(b_layer_inputs) == 1:
                (b_input,) = b_layer_inputs
                in_layers[b_input].extend(bi.output for bi in block_ins)
            elif len(block_ins) > 0:
                raise ValueError(
                    f"A circuit block having multiple inputs cannot be a non-input block"
                )
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
        input_factory: InputLayerFactory,
        sum_factory: SumLayerFactory,
        prod_factory: ProductLayerFactory,
        mixing_factory: Optional[MixingLayerFactory] = None,
        num_channels: int = 1,
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
    ) -> "Circuit":
        layers: List[Layer] = []
        in_layers: Dict[Layer, List[Layer]] = {}
        rgn_to_layers: Dict[RGNode, Layer] = {}

        # Loop through the region graph nodes, which are already sorted in a topological ordering
        for rgn in region_graph.nodes:
            if isinstance(rgn, RegionNode) and not rgn.inputs:  # Input region node
                input_sl = input_factory(rgn.scope, num_input_units, num_channels)
                num_output_units = num_sum_units if rgn.outputs else num_classes
                sum_sl = sum_factory(rgn.scope, num_input_units, num_output_units)
                layers.append(input_sl)
                layers.append(sum_sl)
                in_layers[sum_sl] = [input_sl]
                rgn_to_layers[rgn] = sum_sl
            elif isinstance(rgn, PartitionNode):  # Partition node
                prod_inputs = [rgn_to_layers[rgn_in] for rgn_in in rgn.inputs]
                prod_sl = prod_factory(rgn.scope, num_sum_units, len(prod_inputs))
                layers.append(prod_sl)
                in_layers[prod_sl] = prod_inputs
                rgn_to_layers[rgn] = prod_sl
            elif isinstance(rgn, RegionNode) and len(rgn.inputs) == 1:  # Region node
                num_units = num_sum_units if rgn.outputs else num_classes
                (rgn_in,) = rgn.inputs
                sum_input = rgn_to_layers[rgn_in]
                sum_sl = sum_factory(rgn.scope, sum_input.num_output_units, num_units)
                layers.append(sum_sl)
                in_layers[sum_sl] = [sum_input]
                rgn_to_layers[rgn] = sum_sl
            elif (
                isinstance(rgn, RegionNode) and len(rgn.inputs) > 1
            ):  # Region with multiple partitionings
                if mixing_factory is None:
                    raise ValueError(
                        "A mixing layer factory must be specified to overparameterize multiple region partitionings"
                    )
                num_units = num_sum_units if rgn.outputs else num_classes
                sum_inputs = [rgn_to_layers[rgn_in] for rgn_in in rgn.inputs]
                mix_sl = mixing_factory(rgn.scope, num_units, len(sum_inputs))
                in_layers[mix_sl] = []
                for in_sl in sum_inputs:
                    sum_sl = sum_factory(rgn.scope, in_sl.num_output_units, num_units)
                    layers.append(sum_sl)
                    in_layers[sum_sl] = [in_sl]
                    in_layers[mix_sl].append(sum_sl)
                layers.append(mix_sl)
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
