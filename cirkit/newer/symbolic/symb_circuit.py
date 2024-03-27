from collections import defaultdict, deque
from typing import Dict, Iterable, Iterator, List, Optional, Type

from cirkit.newer.region_graph import RegionGraph
from cirkit.newer.symbolic.layers import (
    SymbInputLayer,
    SymbLayer,
    SymbMixingLayer,
    SymbProdLayer,
    SymbSumLayer,
)
from cirkit.newer.symbolic.symb_op import SymbCircuitOperation
from cirkit.newer.utils import Scope
from cirkit.region_graph import PartitionNode, RegionNode, RGNode


class SymbCircuit:
    """The symbolic representation of a (tensorized) circuit."""

    def __init__(
        self,
        region_graph: RegionGraph,
        layers: Iterable[SymbLayer],
        /,
        *,
        operation: Optional[SymbCircuitOperation] = None,
    ) -> None:
        self.region_graph = region_graph
        self.operation = operation
        self.scope = region_graph.scope
        self.num_vars = region_graph.num_vars
        self.is_smooth = region_graph.is_smooth
        self.is_decomposable = region_graph.is_decomposable
        self.is_structured_decomposable = region_graph.is_structured_decomposable
        self.is_omni_compatible = region_graph.is_omni_compatible
        self._layers = list(layers)

    @classmethod
    def from_region_graph(
        cls,
        region_graph: RegionGraph,
        /,
        *,
        num_channels: int = 1,
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
        input_cls: Optional[Type[SymbInputLayer]] = None,  # TODO: how to specify?
        sum_cls: Optional[Type[SymbSumLayer]] = None,
        prod_cls: Optional[Type[SymbProdLayer]] = None,
    ) -> "SymbCircuit":
        layers: List[SymbLayer] = []
        rgn_to_layers: Dict[RGNode, SymbLayer] = {}

        # Loop through the region graph nodes, which are already sorted in a topological ordering
        for rgn in region_graph.nodes:
            # Retrieve number of units
            if isinstance(rgn, RegionNode):
                num_units = num_sum_units if rgn.outputs else num_classes
            elif isinstance(rgn, PartitionNode):
                num_units = prod_cls.num_prod_units(num_sum_units, len(rgn.inputs))
            else:
                assert False, "Region graph nodes must be either region or partition nodes"

            if isinstance(rgn, RegionNode) and not rgn.inputs:  # Input region node
                input_sl = input_cls(rgn.scope, num_input_units, num_channels)
                sum_sl = sum_cls(rgn.scope, num_units, inputs=[input_sl])
                layers.append(input_sl)
                layers.append(sum_sl)
                rgn_to_layers[rgn] = sum_sl
            elif isinstance(rgn, PartitionNode):  # Partition node
                prod_inputs = [rgn_to_layers[rgn_in] for rgn_in in rgn.inputs]
                prod_sl = prod_cls(rgn.scope, num_units, inputs=prod_inputs)
                layers.append(prod_sl)
                rgn_to_layers[rgn] = prod_sl
            elif isinstance(rgn, RegionNode):  # Inner region node
                sum_inputs = [rgn_to_layers[rgn_in] for rgn_in in rgn.inputs]
                if len(sum_inputs) == 1:  # Region node being partitioned in one way
                    sum_sl = sum_cls(rgn.scope, num_units, inputs=sum_inputs)
                else:  # Region node being partitioned in multiple way -> add "mixing" layer
                    sum_sl = SymbMixingLayer(rgn.scope, num_units, inputs=sum_inputs)
                layers.append(sum_sl)
                rgn_to_layers[rgn] = sum_sl
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "Region graph nodes must be either region or partition nodes"

        return cls(region_graph, layers)

    #######################################    Properties    #######################################
    # Here are the basic properties and some structural properties of the SymbC. Some of them are
    # simply defined in __init__. Some requires additional treatment and is define below. We list
    # everything here to add "docstrings" to them.

    scope: Scope
    """The scope of the SymbC, i.e., the union of scopes of all output layers."""

    num_vars: int
    """The number of variables referenced in the SymbC, i.e., the size of scope."""

    is_smooth: bool
    """Whether the SymbC is smooth, i.e., all inputs to a sum have the same scope."""

    is_decomposable: bool
    """Whether the SymbC is decomposable, i.e., inputs to a product have disjoint scopes."""

    is_structured_decomposable: bool
    """Whether the SymbC is structured-decomposable, i.e., compatible to itself."""

    is_omni_compatible: bool
    """Whether the SymbC is omni-compatible, i.e., compatible to all circuits of the same scope."""

    def is_compatible(
        self, other: "SymbCircuit", /, *, scope: Optional[Iterable[int]] = None
    ) -> bool:
        """Test compatibility with another symbolic circuit over the given scope.

        Args:
            other (SymbCircuit): The other symbolic circuit to compare with.
            scope (Optional[Iterable[int]], optional): The scope over which to check. If None, \
                will use the intersection of the scopes of two SymbC. Defaults to None.

        Returns:
            bool: Whether self is compatible to other.
        """
        return self.region_graph.is_compatible(other.region_graph, scope=scope)

    #######################################    Layer views    ######################################
    # These are iterable views of the nodes in the SymbC, and the topological order is guaranteed
    # (by a stronger ordering). For efficiency, all these views are iterators (implemented as a
    # container iter or a generator), so that they can be chained for iteration without
    # instantiating intermediate containers.

    @property
    def layers(self) -> Iterator[SymbLayer]:
        """All layers in the circuit."""
        return iter(self._layers)

    @property
    def sum_layers(self) -> Iterator[SymbSumLayer]:
        """Sum layers in the circuit, which are always inner layers."""
        return (layer for layer in self.layers if isinstance(layer, SymbSumLayer))

    @property
    def product_layers(self) -> Iterator[SymbProdLayer]:
        """Product layers in the circuit, which are always inner layers."""
        return (layer for layer in self.layers if isinstance(layer, SymbProdLayer))

    @property
    def input_layers(self) -> Iterator[SymbInputLayer]:
        """Input layers of the circuit."""
        return (layer for layer in self.layers if isinstance(layer, SymbInputLayer))

    @property
    def output_layers(self) -> Iterator[SymbSumLayer]:
        """Output layers of the circuit, which are always sum layers."""
        return (layer for layer in self.sum_layers if not layer.outputs)

    @property
    def inner_layers(self) -> Iterator[SymbLayer]:
        """Inner (non-input) layers in the circuit."""
        return (layer for layer in self.layers if layer.inputs)


def pipeline_topological_ordering(pipeline: SymbCircuit) -> List[SymbCircuit]:
    # Initialize the number of incomings edges for each node
    in_symb_circuits: List[SymbCircuit] = []
    num_incomings: Dict[SymbCircuit, int] = defaultdict(int)
    outgoings: Dict[SymbCircuit, List[SymbCircuit]] = defaultdict(list)
    seen, queue = {pipeline}, deque([pipeline])
    while not queue:
        sc = queue.popleft()
        if sc.operation is None:
            in_symb_circuits.append(sc)
            continue
        symb_operands = sc.operation.operands
        for op in symb_operands:
            num_incomings[sc] += 1
            outgoings[op].append(sc)
            if op not in seen:
                seen.add(pipeline)
                queue.append(op)

    # Kahn's algorithm
    ordering: List[SymbCircuit] = []
    to_visit = deque(in_symb_circuits)
    while to_visit:
        sc = to_visit.popleft()
        ordering.append(sc)
        for out_sc in outgoings[sc]:
            num_incomings[out_sc] -= 1
            if num_incomings[out_sc] == 0:
                to_visit.append(out_sc)

    # Check for possible cycles in the pipeline
    for sc, n in num_incomings.keys():
        if n != 0:
            raise ValueError("The given pipeline of symbolic circuit has at least one cycle")
    return ordering
