from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from cirkit.new.layers import InputLayer, Layer, ProductLayer, SumLayer, SumProductLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils import Scope


# Disable: It is designed so.
class SymbolicLayer(ABC):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """The abstract base class for symbolic layers in symbolic circuits."""

    # We accept structure as positional args, and layer spec as kw-only.
    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        scope: Scope,
        layers_in: Iterable["SymbolicLayer"],
        *,
        num_units: int,
        layer_cls: Type[Layer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Construct the SymbolicLayer.

        Args:
            scope (Scope): The scope of this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer, empty for input layers.
            num_units (int): The number of units in this layer.
            layer_cls (Type[Layer]): The concrete layer class to become.
            layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs to initialize \
                layer_cls. Defaults to None.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if layer_cls has no params. Defaults to None.
        """
        super().__init__()
        self.scope = scope

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

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        repr_kv = ", ".join(f"{k}={v}" for k, v in self._repr_dict.items())
        return f"{self.__class__.__name__}@0x{id(self):x}({repr_kv})"

    # We use an abstract instead of direct attribute so that this class includes an abstract method.
    @property
    @abstractmethod
    def _repr_dict(self) -> Dict[str, object]:  # Use object to avoid Any.
        """The dict of key-value pairs used in __repr__."""

    # __hash__ and __eq__ are defined by default to compare on object identity, i.e.,
    # (a is b) <=> (a == b) <=> (hash(a) == hash(b)).


# Disable: It's intended for SymbolicSumLayer to have only these methods.
class SymbolicSumLayer(SymbolicLayer):  # pylint: disable=too-few-public-methods
    """The sum layer in symbolic circuits."""

    # The following attrs have more specific typing.
    layer_cls: Type[Union[SumLayer, SumProductLayer]]
    reparam: Reparameterization  # Sum layer always have params.

    # Note that the typing for layers_in cannot be refined because all layers are mixed in one
    # container in SymbolicCircuit. Same the following two layers.
    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        scope: Scope,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cls: Type[Union[SumLayer, SumProductLayer]],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Reparameterization,
    ) -> None:
        """Construct the SymbolicSumLayer.

        Args:
            scope (Scope): The scope of this layer.
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
            scope,
            layers_in,
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            reparam=reparam,
        )
        assert self.inputs, "SymbolicSumLayer must be an inner layer of the SymbC."

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        # Ignore: Unavoidable for kwargs.
        return {
            "scope": self.scope,
            "arity": self.arity,
            "num_units": self.num_units,
            "layer_cls": self.layer_cls,
            "layer_kwargs": self.layer_kwargs,  # type: ignore[misc]
            "reparam": self.reparam,  # TODO: repr of reparam
        }


# Disable: It's intended for SymbolicProductLayer to have only these methods.
class SymbolicProductLayer(SymbolicLayer):  # pylint: disable=too-few-public-methods
    """The product layer in symbolic circuits."""

    # The following attrs have more specific typing.
    layer_cls: Type[Union[ProductLayer, SumProductLayer]]
    reparam: None  # Product layer has no params.

    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        scope: Scope,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cls: Type[Union[ProductLayer, SumProductLayer]],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Construct the SymbolicProductLayer.

        Args:
            scope (Scope): The scope of this layer.
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
            scope,
            layers_in,
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            reparam=None,
        )

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        # Ignore: Unavoidable for kwargs.
        return {
            "scope": self.scope,
            "arity": self.arity,
            "num_units": self.num_units,
            "layer_cls": self.layer_cls,
            "layer_kwargs": self.layer_kwargs,  # type: ignore[misc]
        }


# Disable: It's intended for SymbolicInputLayer to have only these methods.
class SymbolicInputLayer(SymbolicLayer):  # pylint: disable=too-few-public-methods
    """The input layer in symbolic circuits."""

    # The following attrs have more specific typing.
    layer_cls: Type[InputLayer]
    reparam: Optional[Reparameterization]  # Input layer may or may not have params.

    # Disable: This __init__ is designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        self,
        scope: Scope,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cls: Type[InputLayer],
        layer_kwargs: Optional[Dict[str, Any]] = None,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Construct the SymbolicInputLayer.

        Args:
            scope (Scope): The scope of this layer.
            layers_in (Iterable[SymbolicLayer]): Empty iterable.
            num_units (int): The number of units in this layer.
            layer_cls (Type[InputLayer]): The concrete layer class to become.
            layer_kwargs (Optional[Dict[str, Any]], optional): The additional kwargs to initialize \
                layer_cls. Defaults to None.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if layer_cls has no params. Defaults to None.
        """
        super().__init__(
            scope,
            layers_in,  # Should be empty, will be tested in super().__init__ by its length.
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
            reparam=reparam,
        )
        assert not self.inputs, "SymbolicInputLayer must be an input layer of the SymbC."

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        # Ignore: Unavoidable for kwargs.
        return {
            "scope": self.scope,
            "num_units": self.num_units,
            "layer_cls": self.layer_cls,
            "layer_kwargs": self.layer_kwargs,  # type: ignore[misc]
            "reparam": self.reparam,  # TODO: repr of reparam
        }
