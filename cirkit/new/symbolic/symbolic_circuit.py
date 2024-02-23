from typing import Dict, Iterable, Iterator, Optional, final

import cirkit.new.symbolic.functional as STCF  # SymbolicTensorizedCircuit functional.
from cirkit.new.layers import DenseLayer, InputLayer, MixingLayer, ProductLayer, SumLayer
from cirkit.new.region_graph import PartitionNode, RegionGraph, RegionNode, RGNode
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)
from cirkit.new.utils import OrderedSet, Scope
from cirkit.new.utils.type_aliases import SymbCfgFactory

# TODO: __repr__?


# Mark this class final so that type(SymbC) is always SymbolicTensorizedCircuit.
# DISABLE: It's designed to have these attributes.
@final
# pylint: disable-next=too-many-instance-attributes
class SymbolicTensorizedCircuit:
    """The symbolic representation of a tensorized circuit."""

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        region_graph: RegionGraph,
        /,
        *,
        num_channels: int = 1,
        num_input_units: Optional[int] = None,
        num_sum_units: Optional[int] = None,
        num_classes: int = 1,
        input_cfg: Optional[SymbCfgFactory[InputLayer]] = None,
        sum_cfg: Optional[SymbCfgFactory[SumLayer]] = None,
        prod_cfg: Optional[SymbCfgFactory[ProductLayer]] = None,
        layers: Optional[Iterable[SymbolicLayer]] = None,
    ):
        """Construct the symbolic circuit.

        There are two ways to construct:
            - Provide the number of units and configs to construct symbolic layers from the RG;
            - Provide directly the symbolic layers (must be correctly ordered).

        However in both cases the caller must provide:
            - The RG, which provides the basic properties;
            - The number of channels and classes, which specify the sizes of the input and output.

        If both configs for sum and product specify SumProductLayer as the layer class, they must \
        be the same, which will be used for layer fusion. Otherwise, both should not be \
        SumProductLayer. If directly prodiving layers, the use of SumProductLayer must observe the \
        same rule. This requirement is not checked in symbolic circuit but in TensorizedCircuit.

        Args:
            region_graph (RegionGraph): The underlying region graph.
            num_channels (int, optional): The number of channels of the circuit input, i.e., the \
                number of units for the variables. Defaults to 1.
            num_input_units (Optional[int], optional): The number of units in the input layer. \
                Defaults to None.
            num_sum_units (Optional[int], optional): The number of units in the sum layer. \
                Defaults to None.
            num_classes (int, optional): The number of classes of the circuit output, i.e., the \
                number of units in the output layer. Defaults to 1.
            input_cfg (Optional[SymbCfgFactory[InputLayer]], optional): The config factory for \
                input layers. Defaults to None.
            sum_cfg (Optional[SymbCfgFactory[SumLayer]], optional): The config factory for sum \
                layers. Defaults to None.
            prod_cfg (Optional[SymbCfgFactory[ProductLayer]], optional): The config factory for \
                product layers. Defaults to None.
            layers (Optional[Iterable[SymbolicLayer]], optional): The layers of the circuit, will \
                override all above for layers if provided. Must be ordered in the same way as \
                RGNode but this is not checked. Defaults to None.
        """
        self.region_graph = region_graph
        self.scope = region_graph.scope
        self.num_vars = region_graph.num_vars
        self.is_smooth = region_graph.is_smooth
        self.is_decomposable = region_graph.is_decomposable
        self.is_structured_decomposable = region_graph.is_structured_decomposable
        self.is_omni_compatible = region_graph.is_omni_compatible
        self.num_channels = num_channels
        self.num_classes = num_classes

        if layers is not None:
            self._layers = OrderedSet(layers)
            return

        assert (
            num_input_units is not None
            and num_sum_units is not None
            and input_cfg is not None
            and sum_cfg is not None
            and prod_cfg is not None
        ), "The configs for SymbL to construct SymbC is incomplete."

        # The RGNode and SymbolicLayer does not map 1-to-1 but 1-to-many. This still leads to a
        # deterministic order: SymbolicLayer of different RGNode will be naturally sorted by the
        # RGNode order; SymbolicLayer of the same RGNode are adjcent, and ordered based on the order
        # of edges in the RGNode.
        self._layers = OrderedSet()

        # ANNOTATE: Specify content for empty container.
        node_to_layer: Dict[RGNode, SymbolicLayer] = {}  # Map RGNode to its "output" SymbolicLayer.

        for rg_node in region_graph.nodes:
            if isinstance(rg_node, RegionNode):
                num_units = num_sum_units if rg_node.outputs else num_classes
            elif isinstance(rg_node, PartitionNode):
                num_units = prod_cfg.layer_cls._infer_num_prod_units(
                    num_sum_units, len(rg_node.inputs)
                )
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "This should not happen."

            # Cannot use a generator as layers_in, because it's used twice.
            layers_in = [node_to_layer[node_in] for node_in in rg_node.inputs]
            # ANNOTATE: Different subclasses are assigned below.
            layer_out: SymbolicLayer
            # IGNORE: Unavoidable for kwargs.
            if isinstance(rg_node, RegionNode) and not rg_node.inputs:  # Input region.
                layers_in = [
                    SymbolicInputLayer(
                        rg_node.scope, (), num_units=num_input_units, layer_cfg=input_cfg
                    )
                ]
                # This also works when the input is also output, in which case num_classes is used.
                layer_out = SymbolicSumLayer(
                    rg_node.scope,
                    layers_in,
                    num_units=num_units,
                    layer_cfg=SymbCfgFactory(
                        layer_cls=DenseLayer,
                        layer_kwargs={},  # type: ignore[misc]
                        reparam_factory=sum_cfg.reparam_factory,
                    ),
                )
            elif isinstance(rg_node, RegionNode) and len(rg_node.inputs) == 1:  # Simple inner.
                # layers_in keeps the same.
                layer_out = SymbolicSumLayer(
                    rg_node.scope, layers_in, num_units=num_units, layer_cfg=sum_cfg
                )
            elif isinstance(rg_node, RegionNode) and len(rg_node.inputs) > 1:  # Inner with mixture.
                # MixingLayer cannot change number of units, so must project early.
                layers_in = [
                    SymbolicSumLayer(
                        rg_node.scope, (layer_in,), num_units=num_units, layer_cfg=sum_cfg
                    )
                    for layer_in in layers_in
                ]
                layer_out = SymbolicSumLayer(
                    rg_node.scope,
                    layers_in,
                    num_units=num_sum_units if rg_node.outputs else num_classes,
                    layer_cfg=SymbCfgFactory(
                        layer_cls=MixingLayer,
                        layer_kwargs={},  # type: ignore[misc]
                        reparam_factory=sum_cfg.reparam_factory,
                    ),
                )
            elif isinstance(rg_node, PartitionNode):
                # layers_in keeps the same.
                layer_out = SymbolicProductLayer(
                    rg_node.scope, layers_in, num_units=num_units, layer_cfg=prod_cfg
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
        self, other: "SymbolicTensorizedCircuit", /, *, scope: Optional[Iterable[int]] = None
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
        return (layer for layer in self.layers if isinstance(layer, SymbolicSumLayer))

    @property
    def product_layers(self) -> Iterator[SymbolicProductLayer]:
        """Product layers in the circuit, which are always inner layers."""
        return (layer for layer in self.layers if isinstance(layer, SymbolicProductLayer))

    @property
    def input_layers(self) -> Iterator[SymbolicInputLayer]:
        """Input layers of the circuit."""
        return (layer for layer in self.layers if isinstance(layer, SymbolicInputLayer))

    @property
    def output_layers(self) -> Iterator[SymbolicSumLayer]:
        """Output layers of the circuit, which are always sum layers."""
        return (layer for layer in self.sum_layers if not layer.outputs)

    @property
    def inner_layers(self) -> Iterator[SymbolicLayer]:
        """Inner (non-input) layers in the circuit."""
        return (layer for layer in self.layers if layer.inputs)

    #######################################    Functional    #######################################

    integrate = STCF.integrate
    differentiate = STCF.differentiate
    multiply = STCF.multiply
    mul = STCF.multiply
    __matmul__ = STCF.multiply
    # TODO: __mul__ and __matmul__? first investigate need for "inner product"

    ####################################    (De)Serialization    ###################################
    # TODO: impl? or just save RG and kwargs of SymbC?
