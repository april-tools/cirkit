from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Type

from cirkit.utils.type_aliases import ReparamFactory
from cirkit.reparams.reparam import Reparameterization
from cirkit.reparams.leaf import ReparamIdentity

from cirkit.layers.input.exp_family import (
    ExpFamilyLayer,
    NormalLayer,
    CategoricalLayer,
    BinomialLayer,
)
from cirkit.layers.sum_product import SumProductLayer, TuckerLayer, CPLayer


class SymbolicNode(ABC):
    """Base class for symbolic nodes in symmbolic circuit."""

    inputs: List[Any]
    outputs: List[Any]
    scope: Iterable[int]
    params: Optional[Reparameterization]
    num_output_units: int

    def __init__(self, scope: Iterable[int]) -> None:
        """Construct the Symbolic Node.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        self.scope = frozenset(scope)
        assert self.scope, "The scope of a node must be non-empty"

        self.inputs = []
        self.outputs = []

    def __repr__(self) -> str:
        """Generate the `repr` string of the node."""
        class_name = self.__class__.__name__
        scope = repr(set(self.scope))
        return f"{class_name}:\nScope: {scope}"


class SymbolicSumNode(SymbolicNode):
    """Class representing sum nodes in the symbolic circuit."""

    layer_cls: Type[SumProductLayer]
    layer_kwargs: Optional[Dict[str, Any]]

    def __init__(
        self,
        scope: Iterable[int],
        num_output_units: int,
        layer_cls: Type[SumProductLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the Symbolic Sum Node.

        Args:
            scope (Iterable[int]): The scope of this node.
            num_output_units (int): Number of output units in this node.
            layer_cls (Type[SumProductLayer]): The inner (sum) layer class.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for the inner layer class.
        """
        super().__init__(scope)
        self.num_output_units = num_output_units
        self.layer_cls = layer_cls
        self.layer_kwargs = layer_kwargs

    def set_placeholder_params(
        self,
        num_input_units: int,
        num_output_units: int,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Set un-initialized parameter placeholders for the symbolic sum node.

        Args:
            num_input_units (int): Number of input units.
            num_output_units (int): Number of output units.
            reparam (ReparamFactory): Reparameterization function.
        """
        assert self.num_output_units == num_output_units

        # Handling different layer types
        if self.layer_cls == TuckerLayer:
            self.params = reparam(
                (1, num_input_units, num_input_units, num_output_units), dim=(1, 2)
            )
        elif self.layer_cls == CPLayer:
            raise NotImplementedError("CP Layer not implemented yet")
        # TODO: sum layer (mixing layer)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        layer_cls_name = self.layer_cls.__name__ if self.layer_cls else "None"
        params_shape = getattr(self.params, "shape", None) if hasattr(self, "params") else None

        return (
            f"{class_name}:\n"
            f"Layer Class: {layer_cls_name}\n"
            f"Layer KWArgs: {repr(self.layer_kwargs)}\n"
            f"Output Units: {repr(self.num_output_units)}\n"
            f"Parameter Shape: {repr(params_shape)}"
        )


class SymbolicProductNode(SymbolicNode):
    """Class representing product nodes in the symbolic graph."""

    product_cls: str = "Kroneker Product"  # change to class if we support handaman later

    def __init__(self, scope: Iterable[int], num_input_units: int) -> None:
        """Construct the Symbolic Product Node.

        Args:
            scope (Iterable[int]): The scope of this node.
            num_input_units (int): Number of input units.
        """
        super().__init__(scope)
        # TODO: we only support kroneker product, will we support handaman product?
        if self.product_cls == "Kroneker Product":
            self.num_output_units = num_input_units**2  # Kronecker product output size

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        return (
            f"{class_name}:\n"
            f"Product Class: {repr(self.product_cls)}\n"
            f"Output Units: {repr(self.num_output_units)}"
        )


class SymbolicInputNode(SymbolicNode):
    """Class representing input nodes in the symbolic graph."""

    efamily_cls: Type[ExpFamilyLayer]
    efamily_kwargs: Optional[Dict[str, Any]]

    def __init__(
        self,
        scope: Iterable[int],
        num_output_units: int,
        efamily_cls: Type[ExpFamilyLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the Symbolic Input Node.

        Args:
            scope (Iterable[int]): The scope of this node.
            num_output_units (int): Number of output units.
            efamily_cls (Type[ExpFamilyLayer]): The exponential family class.
            layer_kwargs (Optional[Dict[str, Any]]): The parameters for the exponential family class.
        """
        super().__init__(scope)
        self.num_output_units = num_output_units
        self.efamily_cls = efamily_cls
        self.efamily_kwargs = layer_kwargs

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

        self.params = reparam((1, self.num_output_units, num_replicas, num_suff_stats), dim=-1)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        efamily_cls_name = self.efamily_cls.__name__ if self.efamily_cls else "None"
        params_shape = getattr(self.params, "shape", None) if hasattr(self, "params") else None

        return (
            f"{class_name}:\n"
            f"Input Exp Family Class: {efamily_cls_name}\n"
            f"Layer KWArgs: {repr(self.efamily_kwargs)}\n"
            f"Output Units: {repr(self.num_output_units)}\n"
            f"Parameter Shape: {repr(params_shape)}"
        )
