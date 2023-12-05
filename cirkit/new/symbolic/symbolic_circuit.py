# type: ignore
# pylint: skip-file
from typing import Any, Dict, FrozenSet, Iterable, Optional, Set, Type

from cirkit.layers.input.exp_family import ExpFamilyLayer
from cirkit.layers.sum_product import SumProductLayer
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.rg_node import RGNode
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory


class SymbolicCircuit:
    """The Symbolic Circuit, similar to cirkit.region_graph.RegionGraph."""

    def __init__(
        self,
        region_graph: RegionGraph,
        layer_cls: Type[SumProductLayer],
        efamily_cls: Type[ExpFamilyLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        efamily_kwargs: Optional[Dict[str, Any]] = None,
        reparam: ReparamFactory = ReparamIdentity,
        num_inner_units: int = 2,
        num_input_units: int = 2,
        num_channels: int = 1,
        num_classes: int = 1,
    ):
        """Construct symbolic circuit from a region graph.

        Args:
            region_graph (RegionGraph): The region graph to convert.
            layer_cls (Type[SumProductLayer]): The layer class for inner layers.
            efamily_cls (Type[ExpFamilyLayer]): The layer class for input layers.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for inner layer class.
            efamily_kwargs (Optional[Dict[str, Any]]): The parameters for input layer class.
            reparam (ReparamFactory): The reparametrization function.
            num_inner_units (int): Number of units for inner layers.
            num_input_units (int): Number of units for input layers.
            num_channels (int): Number of channels (e.g., 3 for RGB pixel) for input layers.
            num_classes (int): Number of classes for the PC.

        """
        self.region_graph = region_graph

        self._layers: Set[SymbolicLayer] = set()

        existing_symbolic_layers: Dict[RGNode, SymbolicLayer] = {}

        for input_node in region_graph.input_nodes:
            rg_node_stack = [(input_node, None)]

            while rg_node_stack:
                rg_node, prev_symbolic_layer = rg_node_stack.pop()
                if rg_node in existing_symbolic_layers:
                    symbolic_layer = existing_symbolic_layers[rg_node]
                else:
                    # Construct a symbolic layer from the region node
                    symbolic_layer = self._from_region_node(
                        prev_symbolic_layer,
                        rg_node,
                        layer_cls,
                        efamily_cls,
                        layer_kwargs,
                        efamily_kwargs,
                        reparam,
                        num_inner_units,
                        num_input_units,
                        num_channels,
                        num_classes,
                    )
                    existing_symbolic_layers[rg_node] = symbolic_layer

                # Connect previous symbolic layer to the current one
                if prev_symbolic_layer:
                    self._add_edge(prev_symbolic_layer, symbolic_layer)

                # Handle multiple source nodes
                for output_rg_node in rg_node.outputs:
                    rg_node_stack.append((output_rg_node, symbolic_layer))

    def _from_region_node(
        self,
        prev_symbolic_layer: SymbolicLayer,
        rg_node: RGNode,
        region_graph: RegionGraph,
        layer_cls: Type[SumProductLayer],
        efamily_cls: Type[ExpFamilyLayer],
        layer_kwargs: Optional[Dict[str, Any]],
        efamily_kwargs: Optional[Dict[str, Any]],
        reparam: ReparamFactory,
        num_inner_units: int,
        num_input_units: int,
        num_channels: int,
        num_classes: int,
    ) -> SymbolicLayer:
        """Create a symbolic layer based on the given region node.

        Args:
            prev_symbolic_layer (SymbolicLayer): The parent symbolic layer
            (starting from input layer) that the current layer grown from.
            rg_node (RGNode): The current region graph node to convert to symbolic layer.
            region_graph (RegionGraph): The region graph.
            layer_cls (Type[SumProductLayer]): The layer class for inner layers.
            efamily_cls (Type[ExpFamilyLayer]): The layer class for input layers.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for inner layer class.
            efamily_kwargs (Optional[Dict[str, Any]]): The parameters for input layer class.
            reparam (ReparamFactory): The reparametrization function.
            num_inner_units (int): Number of units for inner layers.
            num_input_units (int): Number of units for input layers.
            num_channels (int): Number of channels (e.g., 3 for RGB pixel) for input layers.
            num_classes (int): Number of classes for the PC.

        Returns:
            SymbolicLayer: The constructed symbolic layer.

        Raises:
            ValueError: If the region node is not valid.
        """
        scope = rg_node.scope
        inputs = rg_node.inputs
        outputs = rg_node.outputs

        if rg_node in region_graph.inner_region_nodes:
            assert len(inputs) == 1, "Inner region nodes should have exactly one input."

            output_units = num_classes if rg_node in region_graph.output_nodes else num_inner_units
            input_units = (
                num_input_units
                if any(
                    isinstance(layer, SymbolicInputLayer) for layer in prev_symbolic_layer.inputs
                )
                else num_inner_units
            )

            symbolic_layer = SymbolicSumLayer(scope, output_units, layer_cls, layer_kwargs)
            symbolic_layer.set_placeholder_params(input_units, output_units, reparam)

        elif rg_node in region_graph.partition_nodes:
            assert len(inputs) == 2, "Partition nodes should have exactly two inputs."
            assert len(outputs) > 0, "Partition nodes should have at least one output."

            left_input_units = num_inner_units if inputs[0].inputs else num_input_units
            right_input_units = num_inner_units if inputs[1].inputs else num_input_units

            assert (
                left_input_units == right_input_units
            ), "Input units for partition nodes should match."

            symbolic_layer = SymbolicProductLayer(scope, left_input_units, layer_cls)

        elif rg_node in region_graph.input_nodes:
            num_replicas = region_graph.num_replicas

            symbolic_layer = SymbolicInputLayer(scope, num_input_units, efamily_cls, efamily_kwargs)
            symbolic_layer.set_placeholder_params(num_channels, num_replicas, reparam)

        else:
            raise ValueError("Region node not valid.")

        return symbolic_layer

    def _add_edge(self, tail: SymbolicLayer, head: SymbolicLayer):
        """Add edge and layer.

        Args:
            tail (SymbolicLayer): The layer the edge originates from.
            head (SymbolicLayer): The layer the edge points to.
        """
        self._layers.add(tail)
        self._layers.add(head)
        tail.outputs.add(head)
        head.inputs.add(tail)

    ##########################    Properties    #########################

    @property
    def scope(self) -> FrozenSet[int]:
        """Get the total scope the circuit."""
        scopes = [layer.scope for layer in self.output_layers]
        return frozenset(set().union(*scopes))

    @property
    def layers(self) -> Iterable[SymbolicLayer]:
        """Get all the layers in the circuit."""
        return iter(self._layers)

    @property
    def input_layers(self) -> Iterable[SymbolicLayer]:
        """Get input layers of the circuiit."""
        return (layer for layer in self.layers if isinstance(layer, SymbolicInputLayer))

    @property
    def output_layers(self) -> Iterable[SymbolicLayer]:
        """Get output layer of the circuit."""
        return (layer for layer in self.layers if not layer.outputs)

    @property
    def sum_layers(self) -> Iterable[SymbolicLayer]:
        """Get inner sum layers of the circuit."""
        return (layer for layer in self.layers if isinstance(layer, SymbolicSumLayer))

    @property
    def product_layers(self) -> Iterable[SymbolicLayer]:
        """Get inner product layers of the circuit."""
        return (layer for layer in self.layers if isinstance(layer, SymbolicProductLayer))

    ##########################    Structural Properties    #########################

    @cached_property
    def is_smooth(self) -> bool:
        """Test smoothness in symbolic circuit."""
        return self.region_graph.is_smooth

    @cached_property
    def is_decomposable(self) -> bool:
        """Test decomposability in symbolic circuit."""
        return self.region_graph.is_decomposable

    @cached_property
    def is_structured_decomposable(self) -> bool:
        """Test structural decomposability in symbolic circuit."""
        return self.region_graph.is_structured_decomposable

    def is_compatible(self, other, x_scope) -> bool:
        """Test compatibility, if self and other are compatible w.r.t x_scope.

        Args:
            other (SymbolicCircuit): Another symbolic circuit to test compatibility.
            x_scope (Iterable[int]): The compatible scope.

        """
        return self.region_graph.is_compatible(other.region_graph, x_scope)
