from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Set, Type

from cirkit.layers.input.exp_family import (
    BinomialLayer,
    CategoricalLayer,
    ExpFamilyLayer,
    NormalLayer,
)
from cirkit.layers.sum_product import (
    CollapsedCPLayer,
    SharedCPLayer,
    SumProductLayer,
    TuckerLayer,
    UncollapsedCPLayer,
)
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.reparams.reparam import Reparameterization
from cirkit.utils.type_aliases import ReparamFactory

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


class SymbolicSumLayer(SymbolicLayer):
    """The sum layer in symbolic circuits."""

    # TODO: how to design interface? require kwargs only?
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        scope: Iterable[int],
        num_units: int,
        layer_cls: Type[SumProductLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the SymbolicSumLayer.

        Args:
            scope (Iterable[int]): The scope of this layer.
            num_units (int): Number of output units in this layer.
            layer_cls (Type[SumProductLayer]): The inner (sum) layer class.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for the inner layer class.

        Raises:
            NotImplementedError: If the shared uncollapsed CP is not implemented.
        """
        super().__init__(scope)
        self.num_units = num_units
        # Ignore: Unavoidable for kwargs.
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}  # type: ignore[misc]
        self.params: Optional[Reparameterization] = None
        self.params_in: Optional[Reparameterization] = None
        self.params_out: Optional[Reparameterization] = None

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

    def set_placeholder_params(
        self,
        num_input_units: int,
        num_units: int,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Set un-initialized parameter placeholders for the symbolic sum layer.

        Args:
            num_input_units (int): Number of input units.
            num_units (int): Number of output units.
            reparam (ReparamFactory): Reparameterization function.

        Raises:
            NotImplementedError: If the shared uncollapsed CP is not implemented.
        """
        assert self.num_units == num_units

        # Handling different layer types
        if self.layer_cls == TuckerLayer:
            # number of fold = 1
            self.params = reparam((1, num_input_units, num_input_units, num_units), dim=(1, 2))
        else:  # CP layer
            # TODO: for unfolded layers we will not need these variants and ignore may be resolved
            arity: int = self.layer_kwargs.get("arity", 2)  # type: ignore[misc]
            assert (
                "fold_mask" not in self.layer_kwargs  # type: ignore[misc]
                or self.layer_kwargs["A"] is None  # type: ignore[misc]
            ), "Do not support fold_mask yet"

            if self.layer_cls == CollapsedCPLayer:
                self.params_in = reparam((1, arity, num_input_units, num_units), dim=-2, mask=None)
            elif self.layer_cls == UncollapsedCPLayer:
                self.params_in = reparam((1, arity, num_input_units, 1), dim=-2, mask=None)
                self.params_out = reparam((1, 1, num_units), dim=-2, mask=None)
            elif self.layer_cls == SharedCPLayer:
                self.params_in = reparam((arity, num_input_units, num_units), dim=-2, mask=None)
            else:
                raise NotImplementedError("The shared uncollapsed CP is not implemented.")

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        class_name = self.__class__.__name__
        layer_cls_name = self.layer_cls.__name__
        # TODO: review this part when we have a new reparams.
        params_shape = self.params.shape if self.params is not None else None

        params_in_shape = self.params_in.shape if self.params_in is not None else None
        params_out_shape = self.params_out.shape if self.params_out is not None else None

        return (
            f"{class_name}:\n"  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            f"Scope: {repr(self.scope)}\n"
            f"Layer Class: {layer_cls_name}\n"
            f"Layer KWArgs: {repr(self.layer_kwargs)}\n"
            f"Number of Units: {repr(self.num_units)}\n"
            f"Parameter Shape: {repr(params_shape)}\n"
            f"CP Layer Parameter in Shape: {repr(params_in_shape)}\n"
            f"CP Layer Parameter out Shape: {repr(params_out_shape)}\n"
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


class SymbolicInputLayer(SymbolicLayer):
    """The input layer in symbolic circuits."""

    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        scope: Iterable[int],
        num_units: int,
        layer_cls: Type[ExpFamilyLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the SymbolicInputLayer.

        Args:
            scope (Iterable[int]): The scope of this layer.
            num_units (int): Number of output units.
            layer_cls (Type[ExpFamilyLayer]): The exponential family class.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for
            the exponential family class.
        """
        # TODO: many things can be merged to SymbolicLayer.__init__.
        super().__init__(scope)
        self.num_units = num_units
        self.layer_cls = layer_cls
        # Ignore: Unavoidable for kwargs.
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}  # type: ignore[misc]
        self.params: Optional[Reparameterization] = None

    def set_placeholder_params(
        self,
        num_channels: int = 1,
        num_replicas: int = 1,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Set un-initialized parameter placeholders for the input layer.

        Args:
            num_channels (int): Number of channels.
            num_replicas (int): Number of replicas.
            reparam (ReparamFactory): Reparameterization function.

        Raises:
            NotImplementedError: Only support Normal, Categorical, and Binomial input layers.
        """
        # Handling different exponential family layer types
        if self.layer_cls == NormalLayer:
            num_suff_stats = 2 * num_channels
        elif self.layer_cls == CategoricalLayer:
            num_suff_stats = (
                self.layer_kwargs["num_categories"] * num_channels  # type: ignore[misc]
            )
        elif self.layer_cls == BinomialLayer:
            num_suff_stats = num_channels
        else:
            raise NotImplementedError("Only support Normal, Categorical, and Binomial input layers")

        self.params = reparam((1, self.num_units, num_replicas, num_suff_stats), dim=-1)

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        class_name = self.__class__.__name__
        efamily_cls_name = self.layer_cls.__name__ if self.layer_cls else "None"
        params_shape = self.params.shape if self.params is not None else None

        return (
            f"{class_name}:\n"  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            f"Scope: {repr(self.scope)}\n"
            f"Input Exp Family Class: {efamily_cls_name}\n"
            f"Layer KWArgs: {repr(self.layer_kwargs)}\n"
            f"Number of Units: {repr(self.num_units)}\n"
            f"Parameter Shape: {repr(params_shape)}\n"
        )
