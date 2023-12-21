import functools
from functools import cached_property
from typing import Dict, Optional, final

import torch
from torch import Tensor, nn

import cirkit.new.model.functional as TCF  # TensorizedCircuit functional.
from cirkit.new.layers import InputLayer, Layer, SumProductLayer
from cirkit.new.symbolic import (
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
    SymbolicTensorizedCircuit,
)


# TODO: this final may not be wanted for user customization, but use of type() in TCF requires it.
# Mark this class final so that type(TensC) is always TensorizedCircuit.
@final
class TensorizedCircuit(nn.Module):
    """The tensorized circuit with concrete computational graph in PyTorch.
    
    This class is aimed for computation, and therefore does not include excessive strutural \
    properties. If those are really needed, use the properties of TensorizedCircuit.symb_circuit.
    """

    def __init__(self, symb_circuit: SymbolicTensorizedCircuit) -> None:
        """Init class.

        Args:
            symb_circuit (SymbolicTensorizedCircuit): The symbolic version of the circuit.
        """
        super().__init__()
        self.symb_circuit = symb_circuit
        self.scope = symb_circuit.scope
        self.num_vars = symb_circuit.num_vars
        self.num_channels = symb_circuit.num_channels
        self.num_classes = symb_circuit.num_classes

        # Automatic nn.Module registry, also in publicly available children names.
        self.layers = nn.ModuleList()

        # The actual internal container for forward, preserves insertion order.
        # ANNOTATE: Specify content for empty container.
        self._symb_to_layers: Dict[SymbolicLayer, Optional[Layer]] = {}

        # Both containers with have a consistent layer order by this loop.
        for symb_layer in symb_circuit.layers:
            # ANNOTATE: Different subclasses are assigned below.
            layer: Optional[Layer]
            # IGNORE: All SymbolicLayer contain Any.
            # IGNORE: Unavoidable for kwargs.
            if issubclass(symb_layer.layer_cls, SumProductLayer) and isinstance(
                symb_layer, SymbolicProductLayer  # type: ignore[misc]
            ):  # Sum-product fusion at prod: build the actual layer with arity of prod.
                # len(symb_layer.outputs) == 1 should be guaranteed by PartitionNode.
                next_layer = symb_layer.outputs[0]  # There should be exactly one SymbSum output.
                assert (
                    isinstance(next_layer, SymbolicSumLayer)  # type: ignore[misc]
                    and next_layer.layer_cls == symb_layer.layer_cls
                ), "Sum-product fusion inconsistent."
                layer = symb_layer.layer_cls(  # TODO: add a function for this cls construction?
                    # TODO: is it good to use only [0]?
                    num_input_units=symb_layer.inputs[0].num_units,
                    num_output_units=next_layer.num_units,
                    arity=symb_layer.arity,
                    reparam=next_layer.reparam,
                    **next_layer.layer_kwargs,  # type: ignore[misc]
                )
            elif issubclass(symb_layer.layer_cls, SumProductLayer) and isinstance(
                symb_layer, SymbolicSumLayer  # type: ignore[misc]
            ):  # Sum-product fusion at sum: just run checks and fill a placeholder.
                prev_layer = symb_layer.inputs[0]  # There should be exactly one SymbProd input.
                assert (
                    symb_layer.arity == 1
                    and isinstance(prev_layer, SymbolicProductLayer)  # type: ignore[misc]
                    and prev_layer.layer_cls == symb_layer.layer_cls
                ), "Sum-product fusion inconsistent."
                layer = None
            elif not issubclass(symb_layer.layer_cls, SumProductLayer):  # Normal layers.
                layer = symb_layer.layer_cls(
                    # TODO: is it good to use only [0]?
                    num_input_units=(  # num_channels for InputLayers or num_units of prev layer.
                        symb_layer.inputs[0].num_units if symb_layer.inputs else self.num_channels
                    ),
                    num_output_units=symb_layer.num_units,
                    arity=(  # Reusing arity to contain num_vars for InputLayer.
                        symb_layer.arity if symb_layer.arity else len(symb_layer.scope)
                    ),
                    reparam=symb_layer.reparam,
                    **symb_layer.layer_kwargs,  # type: ignore[misc]
                )
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "This should not happen."
            if layer is not None:  # Only register actual layers.
                self.layers.append(layer)
            self._symb_to_layers[symb_layer] = layer  # But keep a complete mapping.

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, D, C).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """  # TODO: single letter name?
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    # TODO: do we accept each variable separately?
    def forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, D, C).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """
        # ANNOTATE: Specify content for empty container.
        layer_outputs: Dict[SymbolicLayer, Tensor] = {}  # shape (*B, K).

        for symb_layer, layer in self._symb_to_layers.items():
            if layer is None:
                assert (
                    symb_layer.arity == 1
                ), "Only symbolic layers with arity=1 can be implemented by a place-holder."
                layer_outputs[symb_layer] = layer_outputs[symb_layer.inputs[0]]
                continue

            if isinstance(layer, InputLayer):
                scope_idx = tuple(symb_layer.scope)  # Tensor slice does not accept iterable.
                layer_input = x[..., scope_idx, :].movedim(
                    -2, 0
                )  # shape (*B, D, C) -> (H=D, *B, K=C).
            else:
                layer_input = torch.stack(
                    [layer_outputs[layer_in] for layer_in in symb_layer.inputs], dim=0
                )  # shape H * (*B, K) -> (H, *B, K).
            layer_outputs[symb_layer] = layer(layer_input)

        return torch.stack(
            [layer_outputs[layer_out] for layer_out in self.symb_circuit.output_layers], dim=-2
        )  # shape num_out * (*B, K) -> (*B, num_out, num_cls=K).

    #######################################    Functional    #######################################

    integrate = TCF.integrate

    # Use cached_property to lazily construct the circuit for partition function.
    @cached_property
    def partition_circuit(self) -> "TensorizedCircuit":
        """The circuit calculating the partition function."""
        return self.integrate(scope=self.scope)

    @property
    def partition_func(self) -> Tensor:  # TODO: is this the correct shape?
        """The partition function of the circuit, shape (num_out, num_cls)."""
        # For partition_circuit, the input is irrelevant, so just use zeros.
        # shape (*B, D, C) -> (*B, num_out, num_cls) where *B = ().
        return self.partition_circuit(torch.zeros((self.num_vars, self.num_channels)))

    differentiate = TCF.differentiate

    # NOTE: partialmethod is not suitable here as it does not have __call__ but __get__.
    grad_circuit = cached_property(functools.partial(differentiate, order=1))
    """The circuit calculating the gradient."""

    product = TCF.product
