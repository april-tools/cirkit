# pylint: disable=protected-access
# DISABLE: For this file we disable the above because the functions defined will also be methods of
#          SymbolicTensorizedCircuit, so access to its protected members is expected.

import heapq
import itertools
from typing import TYPE_CHECKING, Dict, Iterable, List, NamedTuple, Optional, Tuple

from cirkit.new.layers import KroneckerLayer, ProdEFLayer
from cirkit.new.reparams import EFProductReparam, KroneckerReparam
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)
from cirkit.new.utils import OrderedSet, Scope
from cirkit.new.utils.type_aliases import SymbLayerCfg

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    from cirkit.new.symbolic.symbolic_circuit import SymbolicTensorizedCircuit


def integrate(
    self: "SymbolicTensorizedCircuit", *, scope: Optional[Iterable[int]] = None
) -> "SymbolicTensorizedCircuit":
    """Integrate the circuit over the variables specified by the given scope.

    Args:
        self (SymbolicTensorizedCircuit): The circuit to integrate.
        scope (Optional[Iterable[int]], optional): The scope over which to integrate, or None for \
            the whole scope of the circuit. Defaults to None.

    Returns:
        SymbolicTensorizedCircuit: The circuit giving the definite integral.
    """
    assert (
        self.is_smooth and self.is_decomposable
    ), "Only smooth and decomposable circuits can be integrated."

    scope = Scope(scope) if scope is not None else self.scope

    integral = object.__new__(type(self))
    # Skip integral.__init__ and customize initialization as below.

    integral.region_graph = self.region_graph
    integral.scope = self.scope  # TODO: is this a good definition?
    integral.num_vars = self.num_vars
    integral.is_smooth = self.is_smooth
    integral.is_decomposable = self.is_decomposable
    integral.is_structured_decomposable = self.is_structured_decomposable
    integral.is_omni_compatible = self.is_omni_compatible
    integral.num_channels = self.num_channels
    integral.num_classes = self.num_classes

    integral._layers = OrderedSet()

    # ANNOTATE: Specify content for empty container.
    self_to_integral: Dict[SymbolicLayer, SymbolicLayer] = {}  # Map between two SymbC.

    for self_layer in self._layers:
        # ANNOTATE: Different subclasses are assigned below.
        integral_layer: SymbolicLayer
        if isinstance(self_layer, SymbolicInputLayer) and self_layer.scope & scope:
            assert (
                self_layer.scope <= scope
            ), "The scope of an input layer must be either all marginalized or all not."
            integral_layer = SymbolicInputLayer(
                self_layer.scope,
                (),
                num_units=self_layer.num_units,
                layer_cfg=self_layer.layer_cls.get_integral(self_layer.layer_cfg),
            )
        else:
            integral_layer = type(self_layer)(
                self_layer.scope,
                (self_to_integral[self_layer_in] for self_layer_in in self_layer.inputs),
                num_units=self_layer.num_units,
                layer_cfg=self_layer.layer_cfg,  # Reuse the same reparam to share params.
            )
        integral._layers.append(integral_layer)
        self_to_integral[self_layer] = integral_layer

    return integral


class _ScopeVarAndSymbLayer(NamedTuple):
    """The tuple of a scope variable and a symbolic layer.

    Used for differential of SymbolicProductLayer.
    """

    scope_var: int  # The id of a variable in the scope of THE SymbolicProductLayer.
    symb_layer: SymbolicProductLayer  # The partial diff of THE SymbolicProductLayer w.r.t. the var.


def differentiate(
    self: "SymbolicTensorizedCircuit", *, order: int = 1
) -> "SymbolicTensorizedCircuit":
    """Differentiate the circuit w.r.t. each variable (i.e. total differentiate) to the given order.

    NOTE: Each output layer will be expanded to (layer.num_vars * num_channels + 1) output layers \
          consecutive in the layer container, with all-but-last layers reshapable to \
          (layer.num_vars, num_channels) calculating the partial differential w.r.t. each variable \
          in the layer's scope and each channel in the variables, and the last one calculating the \
          original, which is copied (but not referenced) from the original circuit, i.e., the same \
          SymbolicLayer object will not be reused in a different SymbolicTensorizedCircuit.

    Args:
        self (SymbolicTensorizedCircuit): The circuit to differentiate.
        order (int, optional): The order of differentiation. Defaults to 1.

    Raises:
        NotImplementedError: When the circuit is not smooth, in which case the differential is too \
            complicated to implement.

    Returns:
        SymbolicTensorizedCircuit: The circuit giving the (total) differential.
    """
    assert self.is_decomposable, "Only decomposable circuits can be differentiated."
    if not self.is_smooth:
        raise NotImplementedError("Differentiation of non-smooth circuit is not yet implemented.")

    assert order >= 0, "The order of differential must be non-negative."
    # TODO: does the following work when order=0? tests?

    differential = object.__new__(type(self))
    # Skip differential.__init__ and customize initialization as below.

    differential.region_graph = self.region_graph
    differential.scope = self.scope  # TODO: is this a good definition?
    differential.num_vars = self.num_vars
    differential.is_smooth = self.is_smooth
    differential.is_decomposable = self.is_decomposable
    differential.is_structured_decomposable = self.is_structured_decomposable
    differential.is_omni_compatible = self.is_omni_compatible
    differential.num_channels = self.num_channels
    differential.num_classes = self.num_classes

    differential._layers = OrderedSet()

    # ANNOTATE: Specify content for empty container.
    self_to_differential: Dict[SymbolicLayer, List[SymbolicLayer]] = {}  # Map between two SymbC.

    for self_layer in self._layers:
        # ANNOTATE: Different subclasses are assigned below.
        differential_layers: List[SymbolicLayer]
        if isinstance(self_layer, SymbolicInputLayer):
            differential_layers = [
                SymbolicInputLayer(
                    self_layer.scope,
                    (),
                    num_units=self_layer.num_units,
                    layer_cfg=self_layer.layer_cls.get_partial(
                        self_layer.layer_cfg, order=order, var_idx=var_idx, ch_idx=ch_idx
                    ),
                )
                for var_idx, ch_idx in itertools.product(
                    range(len(self_layer.scope)), range(self.num_channels)
                )
            ]
        elif isinstance(self_layer, SymbolicSumLayer):
            # Zip to get the layers_in for each of (layer.num_vars * num_channels) partials, except
            # for the copy of self_layer_in at [-1] which will be appended later.
            # TODO: typeshed issue?
            # ANNOTATE: zip gives Any when using *iterables.
            zip_layers_in: Iterable[Tuple[SymbolicLayer, ...]] = zip(
                *(self_to_differential[self_layer_in][:-1] for self_layer_in in self_layer.inputs)
            )
            differential_layers = [  # TODO: use a function to do this transform?
                SymbolicSumLayer(
                    self_layer.scope,
                    layers_in,
                    num_units=self_layer.num_units,
                    layer_cfg=self_layer.layer_cfg,  # Share params in all partial diffs.
                )
                for layers_in in zip_layers_in
            ]
        elif isinstance(self_layer, SymbolicProductLayer):
            # A generator that produces all the partial diffs of self_layer w.r.t. each cur_layer in
            # the input layers of self_layer.
            all_scope_var_symb_layer = (
                # NOTE: The inner level must be a list but not generator, otherwise reference to
                #       locals will be broken.
                # Each element is a list for the partial diffs of self_layer w.r.t. each var and
                # each channel in the scope of the cur_layer.
                [
                    # Each named-tuple is the symb_layer for the partial diff of self_layer w.r.t.
                    # scope_var.
                    _ScopeVarAndSymbLayer(
                        scope_var,
                        SymbolicProductLayer(
                            self_layer.scope,
                            # The inputs to a partial diff is the copy of original input, except for
                            # cur_layer which is replaced by its diff.
                            (
                                diff_cur_layer
                                if self_layer_in == cur_layer
                                else self_to_differential[self_layer_in][-1]
                                for self_layer_in in self_layer.inputs
                            ),
                            num_units=self_layer.num_units,
                            layer_cfg=self_layer.layer_cfg,
                        ),
                    )
                    for var_idx, scope_var in enumerate(cur_layer.scope)
                    for diff_cur_layer in self_to_differential[cur_layer][
                        self.num_channels * var_idx : self.num_channels * (var_idx + 1)
                    ]
                ]
                for cur_layer in self_layer.inputs
            )
            # Each list of the (scope_var, symb_layer) tuples for one cur_layer is sorted by the
            # scope of this cur_layer. And we merge-sort them to get the list of sorted partial
            # diffs by order of scope of self_layer. The channels are consecutive and will be kept
            # in order automatically without explicit sort key for it.
            differential_layers = [
                scope_var_symb_layer.symb_layer
                for scope_var_symb_layer in heapq.merge(
                    *all_scope_var_symb_layer,
                    key=lambda scope_var_symb_layer: scope_var_symb_layer.scope_var,
                )
            ]
        else:
            # NOTE: In the above if/elif, we made all conditions explicit to make it more readable
            #       and also easier for static analysis inside the blocks. Yet the completeness
            #       cannot be inferred and is only guaranteed by larger picture. Also, should
            #       anything really go wrong, we will hit this guard statement instead of going into
            #       a wrong branch.
            assert False, "This should not happen."
        differential_layers.append(  # Append a copy of self_layer.
            type(self_layer)(
                self_layer.scope,
                (self_to_differential[self_layer_in][-1] for self_layer_in in self_layer.inputs),
                num_units=self_layer.num_units,
                layer_cfg=self_layer.layer_cfg,  # Reuse the same reparam to share params.
            )
        )
        differential._layers.extend(differential_layers)
        self_to_differential[self_layer] = differential_layers

    return differential


# TODO: do we use SymbLayerCfg?
# TODO: assert message? also other styling
# TODO: refactor: fixed too complex and too many statements
# pylint: disable-next=too-complex,too-many-statements
def product(
    self: "SymbolicTensorizedCircuit", other: "SymbolicTensorizedCircuit"
) -> "SymbolicTensorizedCircuit":
    """Perform product between two symbolic circuits over the intersected scope.

    Args:
        self (SymbolicTensorizedCircuit): The first circuit to perform product.
        other (SymbolicTensorizedCircuit): The second circuit to perform product.

    Returns:
        SymbolicTensorizedCircuit: The circuit product.
    """
    assert (
        self.is_smooth and self.is_decomposable
    ), "Product could only perform on smooth and decomposable circuits."
    assert (
        other.is_smooth and other.is_decomposable
    ), "Product could only perform on smooth and decomposable circuits."

    scope = self.scope & other.scope
    # assert self.scope == other.scope, "different-scope product is to be implemented."
    assert self.is_compatible(
        other, scope=scope
    ), "Product could only perform on compatible circuits."

    product_circuit = object.__new__(type(self))

    product_circuit.region_graph = self.region_graph  # TODO: implement different-scope product
    # TODO: do we need region graph? region graph is not self!
    product_circuit.scope = self.scope | other.scope
    product_circuit.num_vars = len(product_circuit.scope)
    product_circuit.is_smooth = self.is_smooth
    product_circuit.is_decomposable = self.is_decomposable
    product_circuit.is_structured_decomposable = self.is_structured_decomposable
    # TODO: is_structured_decomposable?
    product_circuit.is_omni_compatible = self.is_omni_compatible and other.is_omni_compatible
    product_circuit.num_channels = self.num_channels
    product_circuit.num_classes = self.num_classes * other.num_classes

    product_circuit._layers = OrderedSet()

    # ANNOTATE: Specify content for empty container.
    # Map between self circuit to product circuit, other circuit to product circuit.
    self_to_product: Dict[SymbolicLayer, SymbolicLayer] = {}
    other_to_product: Dict[SymbolicLayer, SymbolicLayer] = {}

    def _copy_layer(
        circuit: "SymbolicTensorizedCircuit",
        *,
        scope: Scope,
        circuit_is_self: bool,
    ) -> None:
        """Copy layers into the new circuit."""
        for layer in circuit._layers:
            if layer.scope <= scope:
                new_layer = type(layer)(
                    layer.scope,
                    (
                        self_to_product[layer_in] if circuit_is_self else other_to_product[layer_in]
                        for layer_in in layer.inputs
                    ),
                    num_units=layer.num_units,
                    layer_cfg=layer.layer_cfg,  # Reuse the same reparam to share params.
                )
                product_circuit._layers.append(new_layer)
                if circuit_is_self:
                    self_to_product[layer] = new_layer
                else:
                    other_to_product[layer] = new_layer

    def _product(
        self_layer: SymbolicLayer,
        other_layer: SymbolicLayer,
    ) -> SymbolicLayer:
        """Perform product between two layers."""
        assert (
            self_layer.layer_cls == other_layer.layer_cls
        )  # TODO: implement product between cp and tucker
        assert (
            self_layer.layer_cfg.layer_kwargs  # type: ignore[misc]
            == other_layer.layer_cfg.layer_kwargs  # type: ignore[misc]
        )

        new_layer: SymbolicLayer

        # product layer is already generated
        if self_layer in self_to_product and other_layer in other_to_product:
            assert (
                self_to_product[self_layer] is other_to_product[other_layer]
            )  # TODO: different-scope product
            return self_to_product[self_layer]

        if not self_layer.scope & other_layer.scope:
            _copy_layer(self, scope=self_layer.scope, circuit_is_self=True)
            _copy_layer(other, scope=other_layer.scope, circuit_is_self=False)

            new_layer = SymbolicProductLayer(
                Scope(self_layer.scope | other_layer.scope),
                (self_to_product[self_layer], other_to_product[other_layer]),
                num_units=self_layer.num_units,
                # TODO: implement product between circuits with different units
                layer_cfg=SymbLayerCfg(layer_cls=KroneckerLayer),
            )

        elif isinstance(self_layer, SymbolicInputLayer) and isinstance(
            other_layer, SymbolicInputLayer
        ):
            assert self_layer.scope == other_layer.scope, "input layers have different scope"

            if self_layer.layer_cfg.reparam is None or other_layer.layer_cfg.reparam is None:
                raise ValueError("Both layers must have a reparameterization")

            # IGNORE: Unavoidable for kwargs.
            new_layer = SymbolicInputLayer(
                self_layer.scope,
                (),
                num_units=self_layer.num_units * other_layer.num_units,
                # TODO: implement product between circuits with different units
                layer_cfg=SymbLayerCfg(
                    layer_cls=ProdEFLayer,
                    layer_kwargs={  # type: ignore[misc]
                        "ef1_cfg": SymbLayerCfg(
                            layer_cls=self_layer.layer_cls,
                            layer_kwargs=self_layer.layer_cfg.layer_kwargs,  # type: ignore[misc]
                        ),
                        "ef2_cfg": SymbLayerCfg(
                            layer_cls=other_layer.layer_cls,
                            layer_kwargs=other_layer.layer_cfg.layer_kwargs,  # type: ignore[misc]
                        ),
                    },
                    reparam=EFProductReparam(
                        self_layer.layer_cfg.reparam, other_layer.layer_cfg.reparam
                    ),
                ),
            )

        elif isinstance(self_layer, SymbolicSumLayer) and isinstance(other_layer, SymbolicSumLayer):
            self_layer_input = self_layer.inputs
            other_layer_input = other_layer.inputs

            assert (
                len(self_layer_input) == 1 and len(other_layer_input) == 1
            ), "Only 1 input is allowed for sum layers"

            new_layer = SymbolicSumLayer(
                Scope(self_layer.scope | other_layer.scope),
                (_product(self_layer_input[0], other_layer_input[0]),),
                num_units=self_layer.num_units * other_layer.num_units,
                # TODO: implement product between circuits with different units
                layer_cfg=SymbLayerCfg(
                    layer_cls=self_layer.layer_cls,
                    layer_kwargs=self_layer.layer_cfg.layer_kwargs,  # type: ignore[misc]
                    reparam=KroneckerReparam(
                        self_layer.layer_cfg.reparam, other_layer.layer_cfg.reparam
                    ),
                ),
            )

        elif isinstance(self_layer, SymbolicProductLayer) and isinstance(
            other_layer, SymbolicProductLayer
        ):
            self_layer_inputs = self_layer.inputs
            other_layer_inputs = other_layer.inputs

            assert (
                len(self_layer_inputs) == 2 and len(other_layer_inputs) == 2
            ), "product layers only allow for 2 inputs"

            # align the inputs to have the same or similar scope
            aligned_inputs = [
                (self_input, other_input)
                for self_input in self_layer_inputs
                for other_input in other_layer_inputs
                if len(self_input.scope & other_input.scope)
            ]
            self_remaining_inputs = [
                inp
                for inp in self_layer_inputs
                if not any(inp == inputs[0] for inputs in aligned_inputs)
            ]
            other_remaining_inputs = [
                inp
                for inp in other_layer_inputs
                if not any(inp == inputs[1] for inputs in aligned_inputs)
            ]
            aligned_inputs.extend(zip(self_remaining_inputs, other_remaining_inputs))
            # obtain child layers recursively

            new_layer = SymbolicProductLayer(
                Scope(self_layer.scope | other_layer.scope),
                ([_product(pair[0], pair[1]) for pair in aligned_inputs]),
                num_units=self_layer.num_units * other_layer.num_units,
                # TODO: implement product between circuits with different units
                layer_cfg=SymbLayerCfg(
                    layer_cls=self_layer.layer_cls,
                    layer_kwargs=self_layer.layer_cfg.layer_kwargs,  # type: ignore[misc]
                    reparam=None,
                ),
            )

        else:
            raise NotImplementedError("Product with diff scope has not been implemented yet.")

        product_circuit._layers.append(new_layer)
        self_to_product[self_layer] = new_layer
        other_to_product[other_layer] = new_layer
        return new_layer

    # obtain circuit product recursively from the output layers
    for self_layer in self.output_layers:
        for other_layer in other.output_layers:
            _ = _product(self_layer, other_layer)

    return product_circuit


# TODO: refactor SymbC construction? some initial ideas:
#       - still use __init__ to init, but not hack through __new__
#       - use layer factories for construction for each layer (also a last common step?)
#       - self_to_differential[self_layer_in] will be saved and passed to factory
