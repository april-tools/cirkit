from typing import Dict, Optional

import torch
from torch import Tensor, nn

from cirkit.new.layers import InputLayer, Layer, SumProductLayer
from cirkit.new.symbolic import (
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
    SymbolicTensorizedCircuit,
)


class TensorizedCircuit(nn.Module):
    """The tensorized circuit with concrete computational graph in PyTorch.
    
    This class is aimed for computation, and therefore does not include excessive strutural \
    properties. If those are really needed, use the properties of TensorizedCircuit.symb_circuit.
    """

    # TODO: do we also move num_channels to SymbolicTensorizedCircuit?
    def __init__(self, symb_circuit: SymbolicTensorizedCircuit, *, num_channels: int) -> None:
        """Init class.

        All the other config other than num_channels should be provided to the symbolic form.

        Args:
            symb_circuit (SymbolicTensorizedCircuit): The symbolic version of the circuit.
            num_channels (int): The number of channels in the input.
        """
        super().__init__()
        self.symb_circuit = symb_circuit
        self.scope = symb_circuit.scope
        self.num_vars = symb_circuit.num_vars

        self.layers = nn.ModuleList()  # Automatic layer registry, also publically available.

        # TODO: or do we store edges in Layer?
        # The actual internal container for forward.
        self._symb_to_layers: Dict[SymbolicLayer, Optional[Layer]] = {}

        for symb_layer in symb_circuit.layers:
            layer: Optional[Layer]
            # Ignore: all SymbolicLayer contains Any.
            # Ignore: Unavoidable for kwargs.
            if issubclass(symb_layer.layer_cls, SumProductLayer) and isinstance(
                symb_layer, SymbolicProductLayer  # type: ignore[misc]
            ):  # Sum-product fusion at prod: build the actual layer with arity of prod.
                # len(symb_layer.outputs) == 1 should be guaranteed by PartitionNode.
                next_layer = symb_layer.outputs[0]  # There should be exactly one SymbSum output.
                assert (
                    isinstance(next_layer, SymbolicSumLayer)  # type: ignore[misc]
                    and next_layer.layer_cls == symb_layer.layer_cls
                ), "Sum-product fusion inconsistent."
                layer = symb_layer.layer_cls(
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
                prev_layer = symb_layer.inputs[0]  # There should be at exactly SymbProd input.
                assert (
                    len(symb_layer.inputs) == 1  # I.e., symb_layer.arity == 1.
                    and isinstance(prev_layer, SymbolicProductLayer)  # type: ignore[misc]
                    and prev_layer.layer_cls == symb_layer.layer_cls
                ), "Sum-product fusion inconsistent."
                layer = None
            elif not issubclass(symb_layer.layer_cls, SumProductLayer):  # Normal layers.
                layer = symb_layer.layer_cls(
                    # TODO: is it good to use only [0]?
                    num_input_units=(  # num_channels for InputLayers or num_units of prev layer.
                        symb_layer.inputs[0].num_units if symb_layer.inputs else num_channels
                    ),
                    num_output_units=symb_layer.num_units,
                    arity=symb_layer.arity,
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
        # Ignore: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    # TODO: do we accept each variable separately?
    def forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, D, C).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """
        layer_outputs: Dict[SymbolicLayer, Tensor] = {}  # shape (*B, K).

        for symb_layer, layer in self._symb_to_layers.items():
            if layer is None:
                assert (
                    len(symb_layer.inputs) == 1
                ), "Only symbolic layers with arity=1 can be implemented by a place-holder."
                layer_outputs[symb_layer] = layer_outputs[symb_layer.inputs[0]]
                continue

            # Disable: Ternary will be too long for readability.
            if isinstance(layer, InputLayer):  # pylint: disable=consider-ternary-expression
                # TODO: mypy bug? tuple(symb_layer.scope) is inferred to Any
                layer_input = x[..., tuple(symb_layer.scope), :].movedim(  # type: ignore[misc]
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
