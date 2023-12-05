# type: ignore
# pylint: skip-file
from abc import ABC
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
from cirkit.utils.type_aliases import ReparamFactory


class SymbolicLayer(ABC):
    # pylint: disable=too-few-public-methods
    """Base class for symbolic nodes in symmbolic circuit."""

    def __init__(self, scope: Iterable[int]) -> None:
        """Construct the Symbolic Node.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        self.scope = frozenset(scope)
        assert self.scope, "The scope of a node must be non-empty"

        self.inputs: Set[Any] = set()
        self.outputs: Set[Any] = set()

    def __repr__(self) -> str:
        """Generate the `repr` string of the node."""
        class_name = self.__class__.__name__
        scope = repr(set(self.scope))
        return f"{class_name}:\nScope: {scope}\n"


class SymbolicSumLayer(SymbolicLayer):
    """Class representing sum nodes in the symbolic circuit."""

    def __init__(
        self,
        scope: Iterable[int],
        num_units: int,
        layer_cls: Type[SumProductLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the Symbolic Sum Node.

        Args:
            scope (Iterable[int]): The scope of this node.
            num_units (int): Number of output units in this node.
            layer_cls (Type[SumProductLayer]): The inner (sum) layer class.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for the inner layer class.

        Raises:
            NotImplementedError: If the shared uncollapsed CP is not implemented.
        """
        super().__init__(scope)
        self.num_units = num_units
        self.layer_kwargs = layer_kwargs
        self.params = None
        self.params_in = None
        self.params_out = None

        if layer_cls == TuckerLayer:
            self.layer_cls = layer_cls
        else:  # CP layer
            collapsed = (
                self.layer_kwargs["collapsed"] if ("collapsed" in self.layer_kwargs) else True
            )
            shared = self.layer_kwargs["shared"] if ("shared" in self.layer_kwargs) else False

            if not shared and collapsed:
                self.layer_cls = CollapsedCPLayer
            elif not shared and not collapsed:
                self.layer_cls = UncollapsedCPLayer
            elif shared and collapsed:
                self.layer_cls = SharedCPLayer
            else:
                raise NotImplementedError("The shared uncollapsed CP is not implemented.")

    def set_placeholder_params(
        self,
        num_input_units: int,
        num_units: int,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Set un-initialized parameter placeholders for the symbolic sum node.

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
            arity = self.layer_kwargs["arity"] if ("arity" in self.layer_kwargs) else 2
            assert (
                "fold_mask" not in self.layer_kwargs or self.layer_kwargs["A"] is None
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
        """Generate the `repr` string of the node."""
        class_name = self.__class__.__name__
        layer_cls_name = self.layer_cls.__name__ if self.layer_cls else "None"
        params_shape = getattr(self.params, "shape", None) if hasattr(self, "params") else None

        params_in_shape = (
            getattr(self.params_in, "shape", None) if hasattr(self, "params_in") else None
        )
        params_out_shape = (
            getattr(self.params_out, "shape", None) if hasattr(self, "params_out") else None
        )

        return (
            f"{class_name}:\n"
            f"Scope: {repr(self.scope)}\n"
            f"Layer Class: {layer_cls_name}\n"
            f"Layer KWArgs: {repr(self.layer_kwargs)}\n"
            f"Number of Units: {repr(self.num_units)}\n"
            f"Parameter Shape: {repr(params_shape)}\n"
            f"CP Layer Parameter in Shape: {repr(params_in_shape)}\n"
            f"CP Layer Parameter out Shape: {repr(params_out_shape)}\n"
        )


class SymbolicProductLayer(SymbolicLayer):
    # pylint: disable=too-few-public-methods
    """Class representing product nodes in the symbolic graph."""

    def __init__(
        self, scope: Iterable[int], num_units: int, layer_cls: Type[SumProductLayer]
    ) -> None:
        """Construct the Symbolic Product Node.

        Args:
            scope (Iterable[int]): The scope of this node.
            num_units (int): Number of input units.
            layer_cls (Type[SumProductLayer]): The inner (sum) layer class.
        """
        super().__init__(scope)
        self.num_units = num_units
        self.layer_cls = layer_cls

    def __repr__(self) -> str:
        """Generate the `repr` string of the node."""
        class_name = self.__class__.__name__
        layer_cls_name = self.layer_cls.__name__ if self.layer_cls else "None"

        return (
            f"{class_name}:\n"
            f"Scope: {repr(self.scope)}\n"
            f"Layer Class: {layer_cls_name}\n"
            f"Number of Units: {repr(self.num_units)}\n"
        )


class SymbolicInputLayer(SymbolicLayer):
    """Class representing input nodes in the symbolic graph."""

    def __init__(
        self,
        scope: Iterable[int],
        num_units: int,
        efamily_cls: Type[ExpFamilyLayer],
        efamily_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the Symbolic Input Node.

        Args:
            scope (Iterable[int]): The scope of this node.
            num_units (int): Number of output units.
            efamily_cls (Type[ExpFamilyLayer]): The exponential family class.
            efamily_kwargs (Optional[Dict[str, Any]]): The parameters for
            the exponential family class.
        """
        super().__init__(scope)
        self.num_units = num_units
        self.efamily_cls = efamily_cls
        self.efamily_kwargs = efamily_kwargs
        self.params = None

    def set_placeholder_params(
        self,
        num_channels: int = 1,
        num_replicas: int = 1,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Set un-initialized parameter placeholders for the input node.

        Args:
            num_channels (int): Number of channels.
            num_replicas (int): Number of replicas.
            reparam (ReparamFactory): Reparameterization function.

        Raises:
            NotImplementedError: Only support Normal, Categorical, and Binomial input layers.
        """
        # Handling different exponential family layer types
        if self.efamily_cls == NormalLayer:
            num_suff_stats = 2 * num_channels
        elif self.efamily_cls == CategoricalLayer:
            assert "num_categories" in self.efamily_kwargs
            num_suff_stats = self.efamily_kwargs["num_categories"] * num_channels
        elif self.efamily_cls == BinomialLayer:
            num_suff_stats = num_channels
        else:
            raise NotImplementedError("Only support Normal, Categorical, and Binomial input layers")

        self.params = reparam((1, self.num_units, num_replicas, num_suff_stats), dim=-1)

    def __repr__(self) -> str:
        """Generate the `repr` string of the node."""
        class_name = self.__class__.__name__
        efamily_cls_name = self.efamily_cls.__name__ if self.efamily_cls else "None"
        params_shape = getattr(self.params, "shape", None) if hasattr(self, "params") else None

        return (
            f"{class_name}:\n"
            f"Scope: {repr(self.scope)}\n"
            f"Input Exp Family Class: {efamily_cls_name}\n"
            f"Layer KWArgs: {repr(self.efamily_kwargs)}\n"
            f"Number of Units: {repr(self.num_units)}\n"
            f"Parameter Shape: {repr(params_shape)}\n"
        )
