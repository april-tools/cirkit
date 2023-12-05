from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Set, Type

from cirkit.layers.input.exp_family import ExpFamilyLayer
from cirkit.layers.sum_product import (
    CollapsedCPLayer,
    SharedCPLayer,
    SumProductLayer,
    TuckerLayer,
    UncollapsedCPLayer,
)
from cirkit.new.reparams import Reparameterization

# TODO: double check docs and __repr__


# Disable: It's intended for SymbolicLayer to have only these methods.
class SymbolicLayer(ABC):  # pylint: disable=too-few-public-methods
    """The abstract base class for symbolic layers in symbolic circuits."""

    # TODO: Save a RGNode here? allow comparison here?
    def __init__(self, scope: Iterable[int]) -> None:
        """Construct the SymbolicLayer.

        Args:
            scope (Iterable[int]): The scope of this layer.
        """
        super().__init__()
        self.scope = frozenset(scope)
        assert self.scope, "The scope of a layer in SymbC must be non-empty."

        # TODO: should this be a List? what do we need on ordering?
        self.inputs: Set[SymbolicLayer] = set()
        self.outputs: Set[SymbolicLayer] = set()

    # We require subclasses to implement __repr__ on their own. This also forbids the instantiation
    # of this abstract class.
    @abstractmethod
    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """


# Disable: It's intended for SymbolicSumLayer to have only these methods.
class SymbolicSumLayer(SymbolicLayer):  # pylint: disable=too-few-public-methods
    """The sum layer in symbolic circuits."""

    # TODO: how to design interface? require kwargs only?
    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        scope: Iterable[int],
        num_units: int,
        layer_cls: Type[SumProductLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        *,
        reparam: Reparameterization,  # TODO: how to set default here?
    ) -> None:
        """Construct the SymbolicSumLayer.

        Args:
            scope (Iterable[int]): The scope of this layer.
            num_units (int): Number of output units in this layer.
            layer_cls (Type[SumProductLayer]): The inner (sum) layer class.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for the inner layer class.
            reparam (Reparameterization): The reparam.

        Raises:
            NotImplementedError: If the shared uncollapsed CP is not implemented.
        """
        super().__init__(scope)
        self.num_units = num_units
        # Ignore: Unavoidable for kwargs.
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}  # type: ignore[misc]
        self.params = reparam  # TODO: this is not correct, but will be reviewed in new layers.
        self.params_in = reparam
        self.params_out = reparam

        if layer_cls == TuckerLayer:
            self.layer_cls = layer_cls
        else:  # CP layer
            # TODO: for unfolded layers we will not need these variants and ignore may be resolved
            collapsed = layer_kwargs.get("collapsed", True)  # type: ignore[union-attr,misc]
            shared = layer_kwargs.get("shared", False)  # type: ignore[union-attr,misc]

            if not shared and collapsed:  # type: ignore[misc]
                self.layer_cls = CollapsedCPLayer
            elif not shared and not collapsed:  # type: ignore[misc]
                self.layer_cls = UncollapsedCPLayer
            elif shared and collapsed:  # type: ignore[misc]
                self.layer_cls = SharedCPLayer
            else:
                raise NotImplementedError("The shared uncollapsed CP is not implemented.")

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

    def __init__(
        self, scope: Iterable[int], num_units: int, layer_cls: Type[SumProductLayer]
    ) -> None:
        """Construct the SymbolicProductLayer.

        Args:
            scope (Iterable[int]): The scope of this layer.
            num_units (int): Number of input units.
            layer_cls (Type[SumProductLayer]): The inner (sum) layer class.
        """
        super().__init__(scope)
        self.num_units = num_units
        self.layer_cls = layer_cls

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
        scope: Iterable[int],
        num_units: int,
        layer_cls: Type[ExpFamilyLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        *,
        reparam: Reparameterization,  # TODO: how to set default here?
    ) -> None:
        """Construct the SymbolicInputLayer.

        Args:
            scope (Iterable[int]): The scope of this layer.
            num_units (int): Number of output units.
            layer_cls (Type[ExpFamilyLayer]): The exponential family class.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for
                the exponential family class.
            reparam (Reparameterization): The reparam.
        """
        # TODO: many things can be merged to SymbolicLayer.__init__.
        super().__init__(scope)
        self.num_units = num_units
        self.layer_cls = layer_cls
        # Ignore: Unavoidable for kwargs.
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}  # type: ignore[misc]
        self.params = reparam

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        class_name = self.__class__.__name__
        efamily_cls_name = self.layer_cls.__name__ if self.layer_cls else "None"

        return (
            f"{class_name}:\n"  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            f"Scope: {repr(self.scope)}\n"
            f"Input Exp Family Class: {efamily_cls_name}\n"
            f"Layer KWArgs: {repr(self.layer_kwargs)}\n"
            f"Number of Units: {repr(self.num_units)}\n"
        )
