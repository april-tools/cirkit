# pylint: disable=protected-access
# DISABLE: For this file we disable the above because the functions defined will also be methods of
#          SymbolicTensorizedCircuit, so access to its protected members is expected.

import heapq
import itertools
from typing import TYPE_CHECKING, Dict, Iterable, List, NamedTuple, Optional, Tuple

from cirkit.new.layers.inner.inner import InnerLayer
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)
from cirkit.new.utils import OrderedSet, Scope

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
            integral_layer = self_layer.transform(
                (), layer_cfg=self_layer.layer_cls.get_integral(self_layer.layer_cfg)
            )
        else:
            integral_layer = self_layer.transform(
                self_to_integral[self_layer_in] for self_layer_in in self_layer.inputs
            )  # The same reparam shared to integral_layer.
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
                self_layer.transform(
                    (),
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
            # The same reparam shared to all partial diffs.
            differential_layers = [self_layer.transform(layers_in) for layers_in in zip_layers_in]
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
                        # The inputs to a partial diff is the copy of original input, except for
                        # cur_layer which is replaced by its diff.
                        self_layer.transform(
                            diff_cur_layer
                            if self_layer_in == cur_layer
                            else self_to_differential[self_layer_in][-1]
                            for self_layer_in in self_layer.inputs
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
            self_layer.transform(
                self_to_differential[self_layer_in][-1] for self_layer_in in self_layer.inputs
            )
        )
        differential._layers.extend(differential_layers)
        self_to_differential[self_layer] = differential_layers

    return differential


def _product(
    self_layer: SymbolicLayer,
    other_layer: SymbolicLayer,
    pair_to_product: Dict[Tuple[SymbolicLayer, SymbolicLayer], SymbolicLayer],
) -> SymbolicLayer:
    """Perform product between two symbolic layers.

    Args:
        self_layer (SymbolicLayer): The first layer to product (from SymbC self).
        other_layer (SymbolicLayer): The second layer to product (from SymbC other).
        pair_to_product (Dict[Tuple[SymbolicLayer, SymbolicLayer], SymbolicLayer]): The mapping \
            from the pair of layers to their product. The return value is cached here so that we \
            can reuse instead of recalculate.

    Raises:
        NotImplementedError: When "not-yet-implemented feature" is invoked.

    Returns:
        SymbolicLayer: The product of the two layers.
    """
    # This can be above the assert because assertion has already been tested when added.
    if (self_layer, other_layer) in pair_to_product:
        return pair_to_product[(self_layer, other_layer)]

    # DISABLE: We indeed want to compare classes here.
    if type(self_layer) != type(other_layer):  # pylint: disable=unidiomatic-typecheck
        raise NotImplementedError("Product with different RG has not been implemented yet.")
    if (
        issubclass(self_layer.layer_cls, InnerLayer)
        and self_layer.layer_cls != other_layer.layer_cls
    ):
        raise NotImplementedError("Product with different InnerLayer has not been implemented yet.")
    if self_layer.scope != other_layer.scope:
        raise NotImplementedError("Product with different scope has not been implemented yet.")

    if isinstance(self_layer, SymbolicInputLayer):
        # Should be empty. Just use this expr to give a type of Generator[SymbL].
        layers_in = (symbl for symbl in self_layer.inputs)
    elif isinstance(self_layer, SymbolicSumLayer):
        # itertools.product gives the same order of torch.kron, so this automatically maps to the
        # reparam transform of MixingLayer. For DenseLayer there's no ordering because arity=1.
        layers_in = (
            _product(self_layer_in, other_layer_in, pair_to_product)
            for self_layer_in, other_layer_in in itertools.product(
                self_layer.inputs, other_layer.inputs
            )
        )
    elif isinstance(self_layer, SymbolicProductLayer):
        # The order of inputs should be guaranteed to correspond to each other due to the order
        # guaranteed by the layer views.
        layers_in = (
            _product(self_layer_in, other_layer_in, pair_to_product)
            for self_layer_in, other_layer_in in zip(self_layer.inputs, other_layer.inputs)
        )
    else:
        # NOTE: In the above if/elif, we made all conditions explicit to make it more readable
        #       and also easier for static analysis inside the blocks. Yet the completeness
        #       cannot be inferred and is only guaranteed by larger picture. Also, should
        #       anything really go wrong, we will hit this guard statement instead of going into
        #       a wrong branch.
        assert False, "This should not happen."

    # TODO: due to Layer.get_prod
    product_layer = self_layer.transform(
        layers_in,
        num_units=self_layer.num_units * other_layer.num_units,
        layer_cfg=self_layer.layer_cls.get_product(  # type: ignore[misc]
            self_layer.layer_cfg, other_layer.layer_cfg
        ),
    )

    pair_to_product[(self_layer, other_layer)] = product_layer
    return product_layer


def product(
    self: "SymbolicTensorizedCircuit", other: "SymbolicTensorizedCircuit"
) -> "SymbolicTensorizedCircuit":
    """Perform product between two symbolic circuits over the intersected scope.

    Args:
        self (SymbolicTensorizedCircuit): The first circuit to perform product.
        other (SymbolicTensorizedCircuit): The second circuit to perform product.

    Returns:
        SymbolicTensorizedCircuit: The product circuit.
    """
    assert (
        self.is_smooth and self.is_decomposable and other.is_smooth and other.is_decomposable
    ), "Circuits to take product must be smooth and decomposable."
    assert self.is_compatible(other), "Circuits to take product must be compatible to each other."
    assert (
        self.num_channels == other.num_channels
    ), "Circuits to take product must have same channels for input variables."
    # TODO: assert same region graph? compare RG?

    # TODO: tmp disable
    product = object.__new__(type(self))  # pylint: disable=redefined-outer-name
    # Skip product.__init__ and customize initialization as below.

    # TODO: Now we only have same-RG product. may need to be changed later.
    product.region_graph = self.region_graph
    product.scope = self.scope | other.scope
    product.num_vars = len(product.scope)
    product.is_smooth = self.is_smooth
    product.is_decomposable = self.is_decomposable
    # TODO: is the prod struct-decomp?
    product.is_structured_decomposable = self.is_structured_decomposable
    # TODO: is this correct?
    product.is_omni_compatible = self.is_omni_compatible and other.is_omni_compatible
    product.num_channels = self.num_channels
    product.num_classes = self.num_classes * other.num_classes

    # ANNOTATE: Specify content for empty container.
    # Map between (self, other) to product.
    pair_to_product: Dict[Tuple[SymbolicLayer, SymbolicLayer], SymbolicLayer] = {}

    # We must build the layers from outputs because we don't know which layers to product with which
    # if starting from input. We can only know which-to-which by looking as the inputs to a layer
    # pair known to product.
    # TODO: allow "inner product"?
    for self_layer, other_layer in itertools.product(self.output_layers, other.output_layers):
        _product(self_layer, other_layer, pair_to_product)

    # Since we build layers from outputs, we must reorder them to preverse the layer ordering. We
    # reorder by (self_idx, other_idx) so that it's consistent with both the ordering in the two
    # original circuits and the order of kronecker.
    self_idx = {layer: idx for idx, layer in enumerate(self.layers)}
    other_idx = {layer: idx for idx, layer in enumerate(other.layers)}

    # TODO: mypy cannot infer this lambda?
    sorted_pairs = sorted(
        pair_to_product.keys(),
        key=lambda pair: (self_idx[pair[0]], other_idx[pair[1]]),  # type: ignore[misc]
    )

    product._layers = OrderedSet(pair_to_product[pair] for pair in sorted_pairs)

    return product


# TODO: refactor SymbC construction? some initial ideas:
#       - still use __init__ to init, but not hack through __new__
#       - use layer factories for construction for each layer (also a last common step?)
#       - self_to_differential[self_layer_in] will be saved and passed to factory
# TODO: another idea: just optionally provide content of _layer.
