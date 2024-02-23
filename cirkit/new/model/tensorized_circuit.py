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

    def __init__(self, symb_circuit: SymbolicTensorizedCircuit, /) -> None:
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

        # Both containers with have a consistent layer order by this loop.
        for symb_layer in symb_circuit.layers:
            # ANNOTATE: Different subclasses are assigned below.
            layer: Optional[Layer]
            if issubclass(symb_layer.layer_cls, SumProductLayer) and isinstance(
                symb_layer, SymbolicProductLayer
            ):  # Sum-product fusion at prod: just run checks and and give a placeholder.
                # len(symb_layer.outputs) == 1 should be guaranteed by PartitionNode.
                next_layer = symb_layer.outputs[0]  # There should be exactly one SymbSum output.
                assert (
                    isinstance(next_layer, SymbolicSumLayer)
                    and next_layer.layer_cls == symb_layer.layer_cls
                ), "Sum-product fusion inconsistent."
                layer = None
            elif issubclass(symb_layer.layer_cls, SumProductLayer) and isinstance(
                symb_layer, SymbolicSumLayer
            ):  # Sum-product fusion at sum: build the actual layer with fused kwargs.
                prev_layer = symb_layer.inputs[0]  # There should be exactly one SymbProd input.
                assert (
                    symb_layer.arity == 1
                    and isinstance(prev_layer, SymbolicProductLayer)
                    and prev_layer.layer_cls == symb_layer.layer_cls
                ), "Sum-product fusion inconsistent."
                # Concretize based on the config and the output num_units of the SymbolicSumLayer,
                # but the input num_units and the arity of the SymbolicProductLayer.
                layer = symb_layer.concretize(
                    num_input_units=prev_layer.inputs[0].num_units, arity=prev_layer.arity
                )
            elif issubclass(symb_layer.layer_cls, InputLayer):  # Input layers.
                layer = symb_layer.concretize(
                    num_input_units=self.num_channels, arity=len(symb_layer.scope)
                )
            elif not issubclass(symb_layer.layer_cls, SumProductLayer):  # No-fuse inner layers.
                layer = symb_layer.concretize()
            else:
                # NOTE: In the above if/elif, we made all conditions explicit to make it more
                #       readable and also easier for static analysis inside the blocks. Yet the
                #       completeness cannot be inferred and is only guaranteed by larger picture.
                #       Also, should anything really go wrong, we will hit this guard statement
                #       instead of going into a wrong branch.
                assert False, "This should not happen."
            if layer is not None:  # Only register actual layers.
                self.layers.append(layer)

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

        # The graph is only saved in SymbC, so we must loop over SymbLs.
        for symb_layer in self.symb_circuit.layers:
            layer = symb_layer.concrete_layer

            # DISABLE: Extract the assignment for better readability above.
            if layer is None:  # pylint: disable=consider-using-assignment-expr
                # A placeholder, used for SymbolicProductLayer in case of fusion. It's ignored here.
                continue

            if isinstance(layer, InputLayer):
                # Tensor slice does not accept iterable but only sequence.
                # shape (*B, D, C) -> (H=D, *B, K=C).
                layer_input = x[..., tuple(symb_layer.scope), :].movedim(-2, 0)
            else:
                # The placeholder cases are actually handled here. Only arity=1 layers can have
                # a placeholder input.
                layers_in = (
                    symb_layer.inputs
                    if symb_layer.arity > 1 or symb_layer.inputs[0].concrete_layer is not None
                    else symb_layer.inputs[0].inputs
                )
                # TODO: save a copy when arity==1
                # shape H * (*B, K) -> (H, *B, K).
                layer_input = torch.stack(
                    [layer_outputs[layer_in] for layer_in in layers_in], dim=0
                )
            layer_outputs[symb_layer] = layer(layer_input)

        # shape num_out * (*B, K) -> (*B, num_out, num_cls=K).
        return torch.stack(
            [layer_outputs[layer_out] for layer_out in self.symb_circuit.output_layers], dim=-2
        )

    #######################################    Functional    #######################################

    integrate = TCF.integrate

    # Use cached_property to lazily construct the circuit for partition function.
    partition_circuit = cached_property(integrate)
    """The circuit calculating the partition function."""

    @property
    def partition_func(self) -> Tensor:  # TODO: is this the correct shape?
        """The partition function of the circuit, shape (num_out, num_cls)."""
        # For partition_circuit, the input is irrelevant, so just use empty.
        # shape (*B, D, C) -> (*B, num_out, num_cls) where *B = ().
        return self.partition_circuit(torch.empty((self.num_vars, self.num_channels)))

    differentiate = TCF.differentiate

    # NOTE: partialmethod is not suitable here as it does not have __call__ but __get__.
    grad_circuit = cached_property(functools.partial(differentiate, order=1))
    """The circuit calculating the gradient."""

    multiply = TCF.multiply
    mul = TCF.multiply
    __matmul__ = TCF.multiply
    # TODO: __mul__ and __matmul__? match symbolic
