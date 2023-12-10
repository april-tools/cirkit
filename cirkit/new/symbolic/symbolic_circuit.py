from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Type, Union, final

from cirkit.new.layers import (
    DenseLayer,
    InputLayer,
    MixingLayer,
    ProductLayer,
    SumLayer,
    SumProductLayer,
)
from cirkit.new.region_graph import PartitionNode, RegionGraph, RegionNode, RGNode
from cirkit.new.reparams import Reparameterization
from cirkit.new.symbolic.functional import integrate
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)
from cirkit.new.utils import OrderedSet, Scope

# TODO: __repr__?


# Mark this class final so that __class__ of s SymbC is always SymbolicTensorizedCircuit.
# Disable: It's designed to have these many attributes.
@final  # type: ignore[misc]  # Ignore: Caused by kwargs.
class SymbolicTensorizedCircuit:  # pylint: disable=too-many-instance-attributes
    """The symbolic representation of a tensorized circuit."""

    # TODO: is this the best way to provide reparam? or give a layer-wise mapping?
    # TODO: how to design interface? require kwargs only?
    # TODO: how to deal with too-many?
    # pylint: disable-next=too-many-arguments,too-many-locals
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        region_graph: RegionGraph,
        *,
        num_input_units: int,
        num_sum_units: int,
        num_classes: int = 1,
        input_layer_cls: Type[InputLayer],
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        input_reparam: Callable[[], Optional[Reparameterization]] = lambda: None,
        sum_layer_cls: Type[Union[SumLayer, SumProductLayer]],
        sum_layer_kwargs: Optional[Dict[str, Any]] = None,
        sum_reparam: Callable[[], Reparameterization],
        prod_layer_cls: Type[Union[ProductLayer, SumProductLayer]],
        prod_layer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct symbolic circuit from a region graph.

        Args:
            region_graph (RegionGraph): The region graph to convert.
            num_input_units (int): The number of units in the input layer.
            num_sum_units (int): The number of units in the sum layer. Will also be used to infer \
                the number of product units.
            num_classes (int, optional): The number of classes of the circuit output, i.e., the \
                number of units in the output layer. Defaults to 1.
            input_layer_cls (Type[InputLayer]): The layer class for input layers.
            input_layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs for \
                input layer class. Defaults to None.
            input_reparam (Callable[[], Optional[Reparameterization]], optional): The factory to \
                construct reparameterizations for input layer parameters, can produce None if no \
                params is needed. Defaults to lambda: None.
            sum_layer_cls (Type[Union[SumLayer, SumProductLayer]]): The layer class for sum \
                layers, can be either just a class of SumLayer, or a class of SumProductLayer to \
                indicate layer fusion.
            sum_layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs for sum \
                layer class. Defaults to None.
            sum_reparam (Callable[[], Reparameterization]): The factory to construct \
                reparameterizations for sum layer parameters.
            prod_layer_cls (Type[Union[ProductLayer, SumProductLayer]]): The layer class for \
                product layers, can be either just a class of ProductLayer, or a class of \
                SumProductLayer to indicate layer fusion.
            prod_layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs for \
                product layer class, will be ignored if SumProductLayer is used. Defaults to None.
        """
        self.region_graph = region_graph
        self.scope = region_graph.scope
        self.num_vars = region_graph.num_vars
        self.is_smooth = region_graph.is_smooth
        self.is_decomposable = region_graph.is_decomposable
        self.is_structured_decomposable = region_graph.is_structured_decomposable
        self.is_omni_compatible = region_graph.is_omni_compatible
        self.num_classes = num_classes

        self._layers: OrderedSet[SymbolicLayer] = OrderedSet()
        # The RGNode and SymbolicLayer does not map 1-to-1 but 1-to-many. This still leads to a
        # deterministic order: SymbolicLayer of the same RGNode are adjcent, and ordered based on
        # the order of edges in the RG.

        node_to_layer: Dict[RGNode, SymbolicLayer] = {}  # Map RGNode to its "output" SymbolicLayer.

        for rg_node in region_graph.nodes:
            # Cannot use a generator as layers_in, because it's used twice.
            layers_in = [node_to_layer[node_in] for node_in in rg_node.inputs]
            layer_out: SymbolicLayer
            # Ignore: Unavoidable for kwargs.
            if isinstance(rg_node, RegionNode) and not rg_node.inputs:  # Input region.
                layers_in = [
                    SymbolicInputLayer(
                        rg_node,
                        (),  # Old layers_in should be empty.
                        num_units=num_input_units,
                        layer_cls=input_layer_cls,
                        layer_kwargs=input_layer_kwargs,  # type: ignore[misc]
                        reparam=input_reparam(),
                    )
                ]
                # This also works when the input is also output, in which case num_classes is used.
                layer_out = SymbolicSumLayer(
                    rg_node,
                    layers_in,
                    num_units=num_sum_units if rg_node.outputs else num_classes,
                    layer_cls=DenseLayer,  # TODO: can be other sum layer, but how to pass in???
                    layer_kwargs={},  # type: ignore[misc]
                    reparam=sum_reparam(),
                )
            elif isinstance(rg_node, RegionNode) and len(rg_node.inputs) == 1:  # Simple inner.
                # layers_in keeps the same.
                layer_out = SymbolicSumLayer(
                    rg_node,
                    layers_in,
                    num_units=num_sum_units if rg_node.outputs else num_classes,
                    layer_cls=sum_layer_cls,
                    layer_kwargs=sum_layer_kwargs,  # type: ignore[misc]
                    reparam=sum_reparam(),
                )
            elif isinstance(rg_node, RegionNode) and len(rg_node.inputs) > 1:  # Inner with mixture.
                # MixingLayer cannot change number of units, so must project early.
                layers_in = [
                    SymbolicSumLayer(
                        rg_node,
                        (layer_in,),
                        num_units=num_sum_units if rg_node.outputs else num_classes,
                        layer_cls=sum_layer_cls,
                        layer_kwargs=sum_layer_kwargs,  # type: ignore[misc]
                        reparam=sum_reparam(),
                    )
                    for layer_in in layers_in
                ]
                layer_out = SymbolicSumLayer(
                    rg_node,
                    layers_in,
                    num_units=num_sum_units if rg_node.outputs else num_classes,
                    layer_cls=MixingLayer,
                    layer_kwargs={},  # type: ignore[misc]
                    reparam=sum_reparam(),  # TODO: use a constant reparam here?
                )
            elif isinstance(rg_node, PartitionNode):
                # layers_in keeps the same.
                layer_out = SymbolicProductLayer(
                    rg_node,
                    layers_in,
                    num_units=prod_layer_cls._infer_num_prod_units(
                        num_sum_units, len(rg_node.inputs)
                    ),
                    layer_cls=prod_layer_cls,
                    layer_kwargs=prod_layer_kwargs,  # type: ignore[misc]
                    reparam=None,
                )
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "This should not happen."
            # layers_in may be existing layers (from node_layer) which will be de-duplicated by
            # OrderedSet, or newly constructed layers to be added.
            self._layers.extend(layers_in)
            # layer_out is what will be connected to the output of rg_node.
            self._layers.append(layer_out)
            node_to_layer[rg_node] = layer_out

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
        self, other: "SymbolicTensorizedCircuit", *, scope: Optional[Iterable[int]] = None
    ) -> bool:
        """Test compatibility with another symbolic circuit over the given scope.

        Args:
            other (SymbolicCircuit): The other symbolic circuit to compare with.
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
    def layers(self) -> Iterator[SymbolicLayer]:
        """All layers in the circuit."""
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
        # Ignore: SymbolicProductLayer contains Any.
        return (
            layer
            for layer in self.layers
            if isinstance(layer, SymbolicProductLayer)  # type: ignore[misc]
        )

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
        """Output layers of the circuit, which are always sum layers."""
        return (layer for layer in self.sum_layers if not layer.outputs)

    @property
    def inner_layers(self) -> Iterator[SymbolicLayer]:
        """Inner (non-input) layers in the circuit."""
        return (layer for layer in self.layers if layer.inputs)

    #######################################    Functional    #######################################

    integrate = integrate

    ####################################    (De)Serialization    ###################################
    # TODO: impl? or just save RG and kwargs of SymbC?
