from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.utils.algorithms import layerwise_topological_ordering, topological_ordering
from cirkit.utils.scope import Scope


class AbstractTorchCircuit(nn.Module):
    def __init__(
        self,
        scope: Scope,
        num_channels: int,
        layers: List[TorchLayer],
        in_layers: Dict[TorchLayer, List[TorchLayer]],
        out_layers: Dict[TorchLayer, List[TorchLayer]],
        *,
        fold_in_layers_idx: Optional[Dict[TorchLayer, List[List[Tuple[int, int]]]]] = None,
        fold_out_layers_idx: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.scope = scope
        self.num_channels = num_channels
        self._layers: List[TorchLayer] = nn.ModuleList(layers)
        self._in_layers = in_layers
        self._out_layers = out_layers

        # Build up the bookkeeping data structure
        assert (fold_in_layers_idx is None and fold_out_layers_idx is None) or (
            fold_in_layers_idx is not None and fold_out_layers_idx is not None
        )
        if fold_in_layers_idx is None:
            self._bookkeeping = self._build_unfolded_bookkeeping()
        else:
            self._bookkeeping = self._build_folded_bookkeeping(
                fold_in_layers_idx, fold_out_layers_idx
            )

    def _build_unfolded_bookkeeping(self) -> List[Tuple[List[int], Optional[Tensor]]]:
        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # Layer ids
        layer_ids: Dict[TorchLayer, int] = {}

        # Build the bookkeeping data structure
        for l in self._layers:
            if isinstance(l, TorchInputLayer):
                # For input layers, the bookkeeping entry is a tensor index to the input tensor
                bookkeeping_entry = ([], torch.tensor([list(l.scope)]))
            else:
                # For sum/product layers, the bookkeeping entry consists of the indices to the layer inputs
                bookkeeping_entry = ([layer_ids[li] for li in self.layer_inputs(l)], None)
            bookkeeping.append(bookkeeping_entry)
            layer_ids[l] = len(layer_ids)

        # Append a last bookkeeping entry with the info to extract the output tensor
        # This is necessary because we might have circuits with multiple outputs
        output_ids = [layer_ids[lo] for lo in self.output_layers]
        bookkeeping_entry = (output_ids, None)
        bookkeeping.append(bookkeeping_entry)
        return bookkeeping

    def _build_folded_bookkeeping(
        self,
        fold_in_layers_idx: Dict[TorchLayer, List[List[Tuple[int, int]]]],
        fold_out_layers_idx: List[Tuple[int, int]],
    ) -> List[Tuple[List[int], Optional[Tensor]]]:
        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # Get the topological ordering of the layers and build the bookkeeping data structure
        for l in self._layers:
            # Retrieve the index information from the folded layer
            in_layers_idx = fold_in_layers_idx[l]

            if isinstance(l, TorchInputLayer):
                # For input layers, the bookkeeping entry is a tensor index to the input tensor
                in_scope_ids = [[si[1] for si in fi] for fi in in_layers_idx]
                bookkeeping_entry = ([], torch.tensor(in_scope_ids))
            else:
                # Retrieve the unique fold indices that reference the layer inputs
                in_layer_ids = sorted(list(set(si[0] for fi in in_layers_idx for si in fi)))

                # Compute the cumulative indices of the folded inputs
                cum_folded_layer_ids: List[int] = np.cumsum(
                    [0] + [self._layers[li].num_folds for li in in_layer_ids]
                ).tolist()
                cum_folded_layer_ids_map = dict(zip(in_layer_ids, cum_folded_layer_ids))

                # Build the bookkeeping entry
                in_fold_idx: List[List[int]] = []
                for fi in in_layers_idx:
                    in_slice_idx: List[int] = []
                    for si in fi:
                        in_slice_idx.append(cum_folded_layer_ids_map[si[0]] + si[1])
                    in_fold_idx.append(in_slice_idx)
                in_fold_idx_t = torch.tensor(in_fold_idx)
                bookkeeping_entry = (in_layer_ids, in_fold_idx_t)
            bookkeeping.append(bookkeeping_entry)

        # Append a last bookkeeping entry with the info to extract the (possibly multiple) outputs
        out_layers_ids = sorted(list(set(si[0] for si in fold_out_layers_idx)))
        cum_folded_layer_ids: List[int] = np.cumsum(
            [0] + [self._layers[li].num_folds for li in out_layers_ids]
        ).tolist()
        cum_folded_layer_ids_map = dict(zip(out_layers_ids, cum_folded_layer_ids))
        out_fold_idx: List[int] = []
        for si in fold_out_layers_idx:
            out_fold_idx.append(cum_folded_layer_ids_map[si[0]] + si[1])
        out_fold_idx_t = torch.tensor([out_fold_idx])
        bookkeeping_entry = (out_layers_ids, out_fold_idx_t)
        bookkeeping.append(bookkeeping_entry)
        return bookkeeping

    def layer_inputs(self, l: TorchLayer) -> List[TorchLayer]:
        return self._in_layers[l]

    def layer_outputs(self, l: TorchLayer) -> List[TorchLayer]:
        return self._out_layers[l]

    def layers_topological_ordering(self) -> List[TorchLayer]:
        ordering = topological_ordering(
            set(self.output_layers), incomings_fn=lambda l: self._in_layers[l]
        )
        if ordering is None:
            raise ValueError("The given tensorized circuit has at least one layers cycle")
        return ordering

    def layerwise_topological_ordering(self) -> List[List[TorchLayer]]:
        ordering = layerwise_topological_ordering(
            set(self.output_layers), incomings_fn=lambda l: self._in_layers[l]
        )
        if ordering is None:
            raise ValueError("The given tensorized circuit has at least one layers cycle")
        return ordering

    @property
    def layers(self) -> Iterator[TorchLayer]:
        """All layers in the circuit."""
        return iter(self._layers)

    @property
    def input_layers(self) -> Iterator[TorchInputLayer]:
        """Input layers of the circuit."""
        return (layer for layer in self._layers if isinstance(layer, TorchInputLayer))

    @property
    def output_layers(self) -> Iterator[TorchLayer]:
        """Output layers in the circuit."""
        return (layer for layer in self._layers if not self._out_layers[layer])

    def _eval_forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (B, C, D).

        Returns:
            Tensor: The output of the circuit, shape (B, num_classes, K).
        """
        outputs = []  # list of tensors of shape (F, K, B)

        for (in_layer_ids, in_fold_idx), layer in zip(self._bookkeeping[:-1], self._layers):
            # List of input tensors of shape (F', B, K)
            in_tensors = [outputs[i] for i in in_layer_ids]
            if in_tensors:
                if len(in_tensors) > 1:
                    # for it in in_tensors:
                    #    print(type(layer), it.shape)
                    # print()
                    inputs = torch.cat(in_tensors, dim=0)
                else:
                    (inputs,) = in_tensors
                if in_fold_idx is None:
                    inputs = inputs.unsqueeze(dim=0)  # (1, H, B, K)
                else:
                    inputs = inputs[in_fold_idx]  # (F, H, B, K)
            else:
                # The layer is an input layer
                # in_fold_idx has shape (F, D') with D' <= D
                inputs = x[..., in_fold_idx]  # (B, C, F, D')
                # print(in_fold_idx.shape, x.shape, inputs.shape)
                inputs = inputs.permute(2, 1, 0, 3)
            lout = layer(inputs)  # (F, B, K)
            print(type(layer), lout.shape)
            print()
            outputs.append(lout)

        # Retrieve the indices of the output tensors
        out_layer_indices, out_fold_idx = self._bookkeeping[-1]
        # List of tensors of shape (F', B, K)
        out_tensors = [outputs[i] for i in out_layer_indices]
        if len(out_tensors) > 1:
            outputs = torch.cat(out_tensors, dim=0)
        else:
            (outputs,) = out_tensors
        if out_fold_idx is not None:
            outputs = outputs[out_fold_idx].squeeze(dim=0)  # (num_classes, B, K)
        return outputs.transpose(0, 1)  # (B, num_classes, K)


class TorchCircuit(AbstractTorchCircuit):
    """The tensorized circuit with concrete computational graph in PyTorch.

    This class is aimed for computation, and therefore does not include strutural properties.
    """

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
        return self._eval_forward(x)


class TorchConstantCircuit(AbstractTorchCircuit):
    """The tensorized circuit with concrete computational graph in PyTorch.

    This class is aimed for computation, and therefore does not include strutural properties.
    """

    def __call__(self) -> Tensor:
        """Invoke the forward function.

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """  # TODO: single letter name?
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        x = torch.empty(size=(1, self.num_channels, len(self.scope)))
        x = self._eval_forward(x)
        return x.squeeze(dim=0)  # squeeze dummy batch dimension
