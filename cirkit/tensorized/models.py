from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from cirkit.layers import Layer
from cirkit.symbolic.symb_circuit import SymbCircuit


class TensorizedCircuit(nn.Module):
    """The tensorized circuit with concrete computational graph in PyTorch.

    This class is aimed for computation, and therefore does not include excessive strutural \
    properties. If those are really needed, use the properties of TensorizedCircuit.symb_circuit.
    """

    def __init__(
        self,
        symb_circuit: SymbCircuit,
        layers: List[Layer],
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]],
    ) -> None:
        """Init class.

        Args:
            symb_circuit (SymbolicTensorizedCircuit): The symbolic version of the circuit.
        """
        super().__init__()
        self.symb_circuit = symb_circuit
        self.scope = symb_circuit.scope
        self.num_vars = symb_circuit.num_vars

        # Automatic nn.Module registry, also in publicly available children names.
        self.layers = nn.ModuleList(layers)
        self.bookkeeping = bookkeeping

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, D, C).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """  # TODO: single letter name?
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    def forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, D, C).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """
        outputs = []  # list of tensors of shape (K, *B)

        for (in_layer_indices, in_fold_idx), layer in zip(self.bookkeeping, self.layers):
            in_tensors = [
                outputs[i] for i in in_layer_indices
            ]  # list of tensors of shape (*B, K) or empty
            if in_tensors:
                inputs = torch.cat(in_tensors, dim=0)  # (H, *B, K)
            else:  # forward through input layers
                # in_fold_idx has shape (D') with D' <= D
                inputs = x[..., in_fold_idx, :]
                inputs = inputs.permute(0, 1, 2)
            outputs.append(layer(inputs))  # (*B, K)

        out_layer_indices, out_fold_idx = self.bookkeeping[-1]
        out_tensors = [outputs[i] for i in out_layer_indices]  # list of tensors of shape (*B, K)
        outputs = torch.cat(out_tensors, dim=0)
        if out_fold_idx is not None:
            outputs = outputs[out_fold_idx]  # (*B, K)
        return outputs
