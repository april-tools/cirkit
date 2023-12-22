from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from cirkit.new.layers import InputLayer, Layer, ProductLayer, SumLayer, SumProductLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils import Scope


# DISABLE: It's designed to have these attributes and methods.
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class SymbolicLayer(ABC):
    """The abstract base class for symbolic layers in symbolic circuits."""

    # We accept structure as positional args, and layer spec as kw-only.
    # DISABLE: It's designed to have these arguments.
    # IGNORE: Unavoidable for kwargs.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
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
        # another layer's layers_in. No need to de-duplicate, so prefer list over OrderedSet. Both
        # lists automatically gain a consistent ordering with RGNode edge tables by design.
        # ANNOTATE: Specify content for empty container.
        self.inputs: List[SymbolicLayer] = []
        self.outputs: List[SymbolicLayer] = []
        for layer_in in layers_in:
            self.inputs.append(layer_in)
            layer_in.outputs.append(self)

        self.arity = len(self.inputs)
        self.num_units = num_units
        self.layer_cls = layer_cls
        # IGNORE: Unavoidable for kwargs.
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}  # type: ignore[misc]
        self.reparam = reparam

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        repr_kv = ", ".join(f"{k}={v}" for k, v in self._repr_dict.items())
        return f"{type(self).__name__}@0x{id(self):x}({repr_kv})"

    # We use an abstract instead of direct attribute so that this class includes an abstract method.
    # NOTE: Use object to avoid Any. object is sufficient because we only need __repr__.
    @property
    @abstractmethod
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""

    # __hash__ and __eq__ are defined by default to compare on object identity, i.e.,
    # (a is b) <=> (a == b) <=> (hash(a) == hash(b)).


# DISABLE: It's designed to have these methods.
# pylint: disable-next=too-few-public-methods
class SymbolicSumLayer(SymbolicLayer):
    """The sum layer in symbolic circuits."""

    # The following attrs have more specific typing.
    layer_cls: Type[Union[SumLayer, SumProductLayer]]
    reparam: Reparameterization  # Sum layer always have params.

    # Note that the typing for layers_in cannot be refined because all layers are mixed in one
    # container in SymbolicCircuit. Same the following two layers.
    # DISABLE: It's designed to have these arguments.
    # IGNORE: Unavoidable for kwargs.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
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
        # IGNORE: Unavoidable for kwargs.
        super().__init__(
            scope,
            layers_in,
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]
            reparam=reparam,
        )
        assert self.inputs, "SymbolicSumLayer must be an inner layer of the SymbC."

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        # IGNORE: Unavoidable for kwargs.
        return {
            "scope": self.scope,
            "arity": self.arity,
            "num_units": self.num_units,
            "layer_cls": self.layer_cls,
            "layer_kwargs": self.layer_kwargs,  # type: ignore[misc]
            "reparam": self.reparam,  # TODO: repr of reparam
        }


# DISABLE: It's designed to have these methods.
# pylint: disable-next=too-few-public-methods
class SymbolicProductLayer(SymbolicLayer):
    """The product layer in symbolic circuits."""

    # The following attrs have more specific typing.
    layer_cls: Type[Union[ProductLayer, SumProductLayer]]
    reparam: None  # Product layer has no params.

    # DISABLE: It's designed to have these arguments.
    # IGNORE: Unavoidable for kwargs.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
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
        # IGNORE: Unavoidable for kwargs.
        super().__init__(
            scope,
            layers_in,
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]
            reparam=None,
        )

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        # IGNORE: Unavoidable for kwargs.
        return {
            "scope": self.scope,
            "arity": self.arity,
            "num_units": self.num_units,
            "layer_cls": self.layer_cls,
            "layer_kwargs": self.layer_kwargs,  # type: ignore[misc]
        }


# DISABLE: It's designed to have these methods.
# pylint: disable-next=too-few-public-methods
class SymbolicInputLayer(SymbolicLayer):
    """The input layer in symbolic circuits."""

    # The following attrs have more specific typing.
    layer_cls: Type[InputLayer]
    reparam: Optional[Reparameterization]  # Input layer may or may not have params.

    # DISABLE: It's designed to have these arguments.
    # IGNORE: Unavoidable for kwargs.
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
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
        # IGNORE: Unavoidable for kwargs.
        super().__init__(
            scope,
            layers_in,  # Should be empty, will be tested in super().__init__ by its length.
            num_units=num_units,
            layer_cls=layer_cls,
            layer_kwargs=layer_kwargs,  # type: ignore[misc]
            reparam=reparam,
        )
        assert not self.inputs, "SymbolicInputLayer must be an input layer of the SymbC."

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        # IGNORE: Unavoidable for kwargs.
        return {
            "scope": self.scope,
            "num_units": self.num_units,
            "layer_cls": self.layer_cls,
            "layer_kwargs": self.layer_kwargs,  # type: ignore[misc]
            "reparam": self.reparam,  # TODO: repr of reparam
        }
