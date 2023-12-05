from typing import Any, Dict, FrozenSet, Iterable, Iterator, Optional, Set, Type

from cirkit.layers.input.exp_family import ExpFamilyLayer
from cirkit.layers.sum_product import SumProductLayer
from cirkit.new.region_graph import RegionGraph, RGNode
from cirkit.new.reparams import Reparameterization
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)


# Disable: It's designed to have these many attributes.
class SymbolicCircuit:  # pylint: disable=too-many-instance-attributes
    """The Symbolic Circuit."""

    # TODO: how to design interface? require kwargs only?
    # TODO: how to deal with too-many?
    # pylint: disable-next=too-many-arguments,too-many-locals
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        region_graph: RegionGraph,
        layer_cls: Type[SumProductLayer],
        efamily_cls: Type[ExpFamilyLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        efamily_kwargs: Optional[Dict[str, Any]] = None,
        *,
        reparam: Reparameterization,  # TODO: how to set default here?
        num_inner_units: int = 2,
        num_input_units: int = 2,
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
            num_classes (int): Number of classes for the PC.
        """
        self.region_graph = region_graph
        self.scope = region_graph.scope
        self.num_vars = region_graph.num_vars
        self.is_smooth = region_graph.is_smooth
        self.is_decomposable = region_graph.is_decomposable
        self.is_structured_decomposable = region_graph.is_structured_decomposable
        self.is_omni_compatible = region_graph.is_omni_compatible

        self._layers: Set[SymbolicLayer] = set()

        existing_symbolic_layers: Dict[RGNode, SymbolicLayer] = {}

        # TODO: we need to refactor the construction algorithm. better directly assign inputs or
        #       outputs to SymbLayers instead of adding them later.
        # TODO: too many ignores, need to be checked.
        for input_node in region_graph.input_nodes:
            rg_node_stack = [(input_node, None)]

            # TODO: verify this while.
            while rg_node_stack:  # pylint: disable=while-used
                rg_node, prev_symbolic_layer = rg_node_stack.pop()
                if rg_node in existing_symbolic_layers:
                    symbolic_layer = existing_symbolic_layers[rg_node]
                else:
                    # Construct a symbolic layer from the region node
                    symbolic_layer = self._from_region_node(
                        rg_node,
                        region_graph,
                        layer_cls,
                        efamily_cls,
                        layer_kwargs,  # type: ignore[misc]
                        efamily_kwargs,  # type: ignore[misc]
                        reparam,
                        num_inner_units,
                        num_input_units,
                        num_classes,
                    )
                    existing_symbolic_layers[rg_node] = symbolic_layer

                # Connect previous symbolic layer to the current one
                if prev_symbolic_layer:
                    self._add_edge(prev_symbolic_layer, symbolic_layer)  # type: ignore[unreachable]

                # Handle multiple source nodes
                for output_rg_node in rg_node.outputs:
                    rg_node_stack.append((output_rg_node, symbolic_layer))  # type: ignore[arg-type]

    # TODO: the name is not correct. it's not region node.
    # TODO: disable for the moment
    # pylint: disable-next=no-self-use,too-many-arguments,too-many-locals
    def _from_region_node(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        rg_node: RGNode,
        region_graph: RegionGraph,
        layer_cls: Type[SumProductLayer],
        efamily_cls: Type[ExpFamilyLayer],
        layer_kwargs: Optional[Dict[str, Any]],
        efamily_kwargs: Optional[Dict[str, Any]],
        reparam: Reparameterization,
        num_inner_units: int,
        num_input_units: int,
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

        symbolic_layer: SymbolicLayer

        if rg_node in region_graph.inner_region_nodes:  # type: ignore[operator]
            assert len(inputs) == 1, "Inner region nodes should have exactly one input."

            output_units = (
                num_classes
                if rg_node in region_graph.output_nodes  # type: ignore[operator]
                else num_inner_units
            )

            symbolic_layer = SymbolicSumLayer(
                scope, output_units, layer_cls, layer_kwargs, reparam=reparam  # type: ignore[misc]
            )

        elif rg_node in region_graph.partition_nodes:  # type: ignore[operator]
            assert len(inputs) == 2, "Partition nodes should have exactly two inputs."
            assert len(outputs) > 0, "Partition nodes should have at least one output."

            left_input_units = num_inner_units if inputs[0].inputs else num_input_units
            right_input_units = num_inner_units if inputs[1].inputs else num_input_units

            assert (
                left_input_units == right_input_units
            ), "Input units for partition nodes should match."

            symbolic_layer = SymbolicProductLayer(scope, left_input_units, layer_cls)

        elif rg_node in region_graph.input_nodes:  # type: ignore[operator]
            symbolic_layer = SymbolicInputLayer(
                scope,
                num_input_units,
                efamily_cls,
                efamily_kwargs,  # type: ignore[misc]
                reparam=reparam,
            )

        else:
            raise ValueError("Region node not valid.")

        return symbolic_layer

    def _add_edge(self, tail: SymbolicLayer, head: SymbolicLayer) -> None:
        """Add edge and layer.

        Args:
            tail (SymbolicLayer): The layer the edge originates from.
            head (SymbolicLayer): The layer the edge points to.
        """
        self._layers.add(tail)
        self._layers.add(head)
        tail.outputs.add(head)
        head.inputs.add(tail)

    #######################################    Properties    #######################################
    # Here are the basic properties and some structural properties of the SymbC. Some of them are
    # simply defined in __init__. Some requires additional treatment and is define below.
    # We list everything here to add "docstrings" to them.

    scope: FrozenSet[int]
    """The scope of the SymbC, i.e., the union of scopes of all output layers."""

    num_vars: int
    """The number of variables referenced in the SymbC, i.e., the size of scope."""

    is_smooth: bool
    """Whether the SymbC is smooth, i.e., all inputs to a sum have the same scope."""

    is_decomposable: bool
    """Whether the SymbC is decomposable, i.e., inputs to a product have disjoint scopes and their \
    union is the scope of the product."""

    is_structured_decomposable: bool
    """Whether the SymbC is structured-decomposable, i.e., compatible to itself."""

    is_omni_compatible: bool
    """Whether the SymbC is omni-compatible, i.e., compatible to all circuits of the same scope."""

    def is_compatible(
        self, other: "SymbolicCircuit", scope: Optional[Iterable[int]] = None
    ) -> bool:
        """Test compatibility with another symbolic circuit over the given scope.

        Args:
            other (SymbolicCircuit): The other symbolic circuit to compare with.
            scope (Optional[Iterable[int]], optional): The scope over which to check. If None, \
                will use the intersection of the scopes of two SymbC. Defaults to None.

        Returns:
            bool: Whether the SymbC is compatible to the other.
        """
        return self.region_graph.is_compatible(other.region_graph, scope)

    #######################################    Layer views    ######################################
    # These are iterable views of the nodes in the SymbC. For efficiency, all these views are
    # iterators (implemented as a container iter or a generator), so that they can be chained for
    # iteration without instantiating intermediate containers.
    # NOTE: There's no ordering graranteed for these views. However RGNode can be sorted if needed.

    @property
    def layers(self) -> Iterator[SymbolicLayer]:
        """All the layers in the circuit."""
        return iter(self._layers)

    @property
    def sum_layers(self) -> Iterator[SymbolicSumLayer]:
        """Sum layers in the circuit, which are always inner layers."""
        # Ignore: SymbolicSumLayer contains Any.
        return (
            layer
            for layer in self.layers
            if isinstance(layer, SymbolicSumLayer)  # type: ignore[misc]
        )

    @property
    def product_layers(self) -> Iterator[SymbolicProductLayer]:
        """Product layers in the circuit, which are always inner layers."""
        return (layer for layer in self.layers if isinstance(layer, SymbolicProductLayer))

    @property
    def input_layers(self) -> Iterator[SymbolicInputLayer]:
        """Input layers of the circuit."""
        # Ignore: SymbolicInputLayer contains Any.
        return (
            layer
            for layer in self.layers
            if isinstance(layer, SymbolicInputLayer)  # type: ignore[misc]
        )

    @property
    def output_layers(self) -> Iterator[SymbolicSumLayer]:
        """Output layer of the circuit, which are guaranteed to be sum layers."""
        return (layer for layer in self.sum_layers if not layer.outputs)

    ####################################    (De)Serialization    ###################################
    # TODO: impl?
