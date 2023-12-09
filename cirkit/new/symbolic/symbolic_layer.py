from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from cirkit.new.layers import InputLayer, Layer, ProductLayer, SumLayer, SumProductLayer
from cirkit.new.region_graph import PartitionNode, RegionNode, RGNode
from cirkit.new.reparams import Reparameterization

# TODO: double check __repr__


# Disable: It's intended for SymbolicLayer to have these many attrs.
class SymbolicLayer(ABC):  # pylint: disable=too-many-instance-attributes
    """The abstract base class for symbolic layers in symbolic circuits."""

    # We accept structure as positional args, and layer spec as kw-only.
    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        rg_node: RGNode,
        layers_in: Iterable["SymbolicLayer"],
        *,
        num_units: int,
        layer_cls: Type[Layer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Construct the SymbolicLayer.

        Args:
            rg_node (RGNode): The region graph node corresponding to this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer, empty for input layers.
            num_units (int): The number of units in this layer.
            layer_cls (Type[Layer]): The concrete layer class to become.
            layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs to initialize \
                layer_cls. Defaults to None.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if layer_cls has no params. Defaults to None.
        """
        super().__init__()
        self.rg_node = rg_node
        self.scope = rg_node.scope

        # self.inputs is filled using layers_in, while self.outputs is empty until self appears in
        # another layer's layers_in. No need to de-duplicate, so prefer list over OrderedSet.
        self.inputs: List[SymbolicLayer] = []
        self.outputs: List[SymbolicLayer] = []
        for layer_in in layers_in:
            self.inputs.append(layer_in)
            layer_in.outputs.append(self)

        self.arity = len(self.inputs) if self.inputs else 1  # InputLayer is defined with artiy=1.
        self.num_units = num_units
        self.layer_cls = layer_cls
        # Ignore: Unavoidable for kwargs.
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}  # type: ignore[misc]
        self.reparam = reparam

    # We require subclasses to implement __repr__ on their own. This also forbids the instantiation
    # of this abstract class.
    @abstractmethod
    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """

    # __hash__ and __eq__ are defined by default to compare on object identity, i.e.,
    # (a is b) <=> (a == b) <=> (hash(a) == hash(b)).

    def __lt__(self, other: "SymbolicLayer") -> bool:
        """Compare the layer with another layer, for < operator implicitly used in sorting.

        SymbolicLayer is compared by the corresponding RGNode, so that SymbolicCircuit obtains the \
        same ordering as the RegionGraph.

        Args:
            other (SymbolicLayer): The other layer to compare with.

        Returns:
            bool: Whether self < other.
        """
        return (
            self.rg_node < other.rg_node or self.rg_node == other.rg_node and self in other.outputs
        )  # Either the corresponding rg_node precedes, or for same rg_node, self directly precedes.


# Disable: It's intended for SymbolicSumLayer to have only these methods.
class SymbolicSumLayer(SymbolicLayer):  # pylint: disable=too-few-public-methods
    """The sum layer in symbolic circuits."""

    reparam: Reparameterization  # Sum layer always have params.

    # Note that the typing for layers_in cannot be refined because all layers are mixed in one
    # container in SymbolicCircuit. Same the following two layers.
    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        rg_node: RegionNode,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cls: Type[Union[SumLayer, SumProductLayer]],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Reparameterization,
    ) -> None:
        """Construct the SymbolicSumLayer.

        Args:
            rg_node (RegionNode): The region node corresponding to this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer.
            num_units (int): The number of units in this layer.
            layer_cls (Type[Union[SumLayer, SumProductLayer]]): The concrete layer class to \
                become, can be either just a class of SumLayer, or a class of SumProductLayer to \
                indicate layer fusion.
            layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs to initialize \
                layer_cls. Defaults to None.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        super().__init__(
            rg_node,
            layers_in,
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            reparam=reparam,
        )
        assert self.inputs, "SymbolicSumLayer must be an inner layer of the SymbC."

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        class_name = self.__class__.__name__
        layer_cls_name = self.layer_cls.__name__

        return (
            f"{class_name}:\n"  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            f"Scope: {repr(self.scope)}\n"
            f"Layer Class: {layer_cls_name}\n"
            f"Layer KWArgs: {repr(self.layer_kwargs)}\n"
            f"Number of Units: {repr(self.num_units)}\n"
        )


# Disable: It's intended for SymbolicProductLayer to have only these methods.
class SymbolicProductLayer(SymbolicLayer):  # pylint: disable=too-few-public-methods
    """The product layer in symbolic circuits."""

    reparam: None  # Product layer has no params.

    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        rg_node: PartitionNode,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cls: Type[Union[ProductLayer, SumProductLayer]],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Construct the SymbolicProductLayer.

        Args:
            rg_node (PartitionNode): The partition node corresponding to this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer.
            num_units (int): The number of units in this layer.
            layer_cls (Type[Union[ProductLayer, SumProductLayer]]): The concrete layer class to \
                become, can be either just a class of ProductLayer, or a class of SumProductLayer \
                to indicate layer fusion.
            layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs to initialize \
                layer_cls. Defaults to None.
            reparam (Optional[Reparameterization], optional): Ignored. This layer has no params. \
                Defaults to None.
        """
        super().__init__(
            rg_node,
            layers_in,
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            reparam=None,
        )

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        class_name = self.__class__.__name__
        layer_cls_name = self.layer_cls.__name__

        return (
            f"{class_name}:\n"
            f"Scope: {repr(self.scope)}\n"
            f"Layer Class: {layer_cls_name}\n"
            f"Number of Units: {repr(self.num_units)}\n"
        )


# Disable: It's intended for SymbolicInputLayer to have only these methods.
class SymbolicInputLayer(SymbolicLayer):  # pylint: disable=too-few-public-methods
    """The input layer in symbolic circuits."""

    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        rg_node: RegionNode,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cls: Type[InputLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Construct the SymbolicInputLayer.

        Args:
            rg_node (RegionNode): The region node corresponding to this layer.
            layers_in (Iterable[SymbolicLayer]): Empty iterable.
            num_units (int): The number of units in this layer.
            layer_cls (Type[InputLayer]): The concrete layer class to become.
            layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs to initialize \
                layer_cls. Defaults to None.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if layer_cls has no params. Defaults to None.
        """
        super().__init__(
            rg_node,
            layers_in,  # Should be empty, will be tested in super().__init__ by its length.
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            reparam=reparam,
        )
        assert not self.inputs, "SymbolicInputLayer must be an input layer of the SymbC."

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        class_name = self.__class__.__name__
        layer_cls_name = self.layer_cls.__name__ if self.layer_cls else "None"

        return (
            f"{class_name}:\n"  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            f"Scope: {repr(self.scope)}\n"
            f"Input Exp Family Class: {layer_cls_name}\n"
            f"Layer KWArgs: {repr(self.layer_kwargs)}\n"
            f"Number of Units: {repr(self.num_units)}\n"
        )
