from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from cirkit.backend.torch.layers import TorchLayer


class TensorizedCircuit(nn.Module):
    """The tensorized circuit with concrete computational graph in PyTorch.

    This class is aimed for computation, and therefore does not include strutural properties.
    """

    def __init__(
        self,
        layers: List[TorchLayer],
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.bookkeeping = bookkeeping

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, C, D).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """  # TODO: single letter name?
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    def forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (B, C, D).

        Returns:
            Tensor: The output of the circuit, shape (B, num_classes, K).
        """
        outputs = []  # list of tensors of shape (F, K, B)

        for (in_layer_indices, in_fold_idx), layer in zip(self.bookkeeping, self.layers):
            # List of input tensors of shape (B, K) or empty [without folding]
            # List of input tensors of shape (F, B, K) or empty [with folding]
            # If non-empty and if the layer is not folded, then F = 1
            in_tensors = [outputs[i] for i in in_layer_indices]

            if in_tensors:
                # The layer is not an input layer
                # Bypass cat if there is only one input tensor to this layer
                if len(in_tensors) > 1:
                    inputs = torch.stack(in_tensors, dim=0)
                else:
                    (inputs,) = in_tensors

                if in_fold_idx is None:
                    # This layer is not folded, introduce fold dimension
                    # inputs = inputs.unsqueeze(dim=0)  # (1, H, B, K) [with folding]
                    pass  # inputs: (H, B, K) [without folding]
                else:
                    # This layer is folded, and in_fold_idx has shape (F, H)
                    inputs = x[in_fold_idx]  # (F', H, B, K) [with folding]
                    assert False  # [without folding]
            else:
                # The layer is an input layer
                # in_fold_idx has shape (D',) with D' <= D  [without folding]
                inputs = x[..., in_fold_idx.squeeze(dim=0)].transpose(
                    0, 1
                )  # (C, B, D)  [without folding]

            # outputs.append(layer(inputs))  # (F', B, K), with possibly F' = 1 [with folding]
            outputs.append(layer(inputs))  # (B, K) [without folding]

        # Retrieve the indices of the output tensors
        out_layer_indices, out_fold_idx = self.bookkeeping[-1]
        # out_tensors = [outputs[i] for i in out_layer_indices]  # list of tensors of shape (F, B, K)  [with folding]
        out_tensors = [
            outputs[i] for i in out_layer_indices
        ]  # list of tensors of shape (B, K)  [without folding]

        # Bypass cat if there is only one output tensor
        if len(out_tensors) > 1:
            outputs = torch.stack(out_tensors, dim=0)
        else:
            (outputs,) = out_tensors
            outputs = outputs.unsqueeze(dim=0)
        if out_fold_idx is not None:
            outputs = outputs[out_fold_idx]  # (num_classes, B, K)

        # Move batch dimension to be the first dimension
        return outputs.permute(1, 2, 0)  # (B, F', K)
