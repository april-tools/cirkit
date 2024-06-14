from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph import TorchDiAcyclicGraph, build_unfolded_address_book
from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
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
        topologically_ordered: bool = False,
        in_fold_idx: Optional[Dict[TorchLayer, List[List[Tuple[int, int]]]]] = None,
        out_fold_idx: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self._graph = TorchDiAcyclicGraph[TorchLayer](layers, in_layers, out_layers, topologically_ordered=topologically_ordered)
        self.scope = scope
        self.num_channels = num_channels

        # Build the bookkeeping data structure
        assert (in_fold_idx is None and out_fold_idx is None) or (
                in_fold_idx is not None and out_fold_idx is not None
        )
        if in_fold_idx is None:
            self._bookkeeping = self._build_unfolded_bookkeeping()
        else:
            self._bookkeeping = self._build_folded_bookkeeping(
                in_fold_idx, out_fold_idx
            )

    @property
    def is_topologically_ordered(self) -> bool:
        return self._graph.is_topologically_ordered

    def layer_inputs(self, l: TorchLayer) -> List[TorchLayer]:
        return self._graph.node_inputs(l)

    def node_outputs(self, l: TorchLayer) -> List[TorchLayer]:
        return self._graph.node_outputs(l)

    @property
    def layers(self) -> List[TorchLayer]:
        return self._graph.nodes

    @property
    def layers_inputs(self) -> Dict[TorchLayer, List[TorchLayer]]:
        return self._graph.nodes_inputs

    @property
    def layers_outputs(self) -> Dict[TorchLayer, List[TorchLayer]]:
        return self._graph.nodes_outputs

    @property
    def inputs(self) -> Iterator[TorchInputLayer]:
        return (l for l in self._graph.inputs if isinstance(l, TorchInputLayer))

    @property
    def outputs(self) -> Iterator[TorchLayer]:
        return self._graph.outputs

    def topological_ordering(self) -> Iterator[TorchLayer]:
        return self._graph.topological_ordering()

    def layerwise_topological_ordering(self) -> Iterator[List[TorchLayer]]:
        return self._graph.layerwise_topological_ordering()

    def _build_unfolded_bookkeeping(self) -> List[Tuple[List[int], Optional[Tensor]]]:
        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # Layer ids
        layer_ids: Dict[TorchLayer, int] = {}

        # Build the bookkeeping data structure
        for l in self.topological_ordering():
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
        out_layers_ids = [layer_ids[lo] for lo in self.outputs]
        bookkeeping_entry = (out_layers_ids, None)
        bookkeeping.append(bookkeeping_entry)
        return bookkeeping

    def _build_folded_bookkeeping(
        self,
        in_fold_idx: Dict[TorchLayer, List[List[Tuple[int, int]]]],
        out_fold_idx: List[Tuple[int, int]],
    ) -> List[Tuple[List[int], Optional[Tensor]]]:
        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # A useful data dictonary mapping layer ids to their number of folds
        num_folds_map: Dict[int, int] = {}

        # Build the bookkeeping data structure
        for l in self.topological_ordering():
            # Retrieve the index information from the folded layer
            in_layers_idx = in_fold_idx[l]

            if isinstance(l, TorchInputLayer):
                # For input layers, the bookkeeping entry is a tensor index to the input tensor
                in_scope_ids = [[si[1] for si in fi] for fi in in_layers_idx]
                bookkeeping_entry = ([], torch.tensor(in_scope_ids))
            else:
                # Retrieve the unique fold indices that reference the layer inputs
                bk_in_layer_ids = sorted(list(set(si[0] for fi in in_layers_idx for si in fi)))

                # Compute the cumulative indices of the folded inputs
                cum_folded_layer_ids_map = dict(
                    zip(
                        bk_in_layer_ids,
                        np.cumsum([0] + [num_folds_map[li] for li in bk_in_layer_ids]).tolist(),
                    )
                )

                # Build the bookkeeping entry
                bk_in_fold_idx: List[List[int]] = []
                for fi in in_layers_idx:
                    in_slice_idx: List[int] = []
                    for si in fi:
                        in_slice_idx.append(cum_folded_layer_ids_map[si[0]] + si[1])
                    bk_in_fold_idx.append(in_slice_idx)
                bk_in_fold_idx_t = torch.tensor(bk_in_fold_idx)
                bookkeeping_entry = (bk_in_layer_ids, bk_in_fold_idx_t)
            num_folds_map[len(bookkeeping)] = l.num_folds
            bookkeeping.append(bookkeeping_entry)

        # Append a last bookkeeping entry with the info to extract the (possibly multiple) outputs
        out_layers_ids = sorted(list(set(si[0] for si in out_fold_idx)))
        cum_folded_layer_ids_map = dict(
            zip(
                out_layers_ids,
                np.cumsum([0] + [num_folds_map[li] for li in out_layers_ids]).tolist(),
            )
        )
        out_fold_idx: List[int] = [
            cum_folded_layer_ids_map[si[0]] + si[1] for si in out_fold_idx
        ]
        if out_fold_idx == list(range(len(out_fold_idx))):
            out_fold_idx_t = None
        else:
            out_fold_idx_t = torch.tensor([out_fold_idx])
        bookkeeping_entry = (out_layers_ids, out_fold_idx_t)
        bookkeeping.append(bookkeeping_entry)
        return bookkeeping

    def _eval_forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (B, C, D).

        Returns:
            Tensor: The output of the circuit, shape (B, num_classes, K).
        """
        outputs = []  # list of tensors of shape (F, K, B)

        ordering = self.topological_ordering()
        for (in_layer_ids, in_fold_idx), layer in zip(self._bookkeeping[:-1], ordering):
            # List of input tensors of shape (F', B, K)
            in_tensors = [outputs[i] for i in in_layer_ids]
            if in_tensors:
                if len(in_tensors) > 1:
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
                inputs = inputs.permute(2, 1, 0, 3)
            lout = layer(inputs)  # (F, B, K)
            outputs.append(lout)

        # Retrieve the indices of the output tensors
        out_layers_ids, out_fold_idx = self._bookkeeping[-1]
        # List of tensors of shape (F', B, K)
        out_tensors = [outputs[i] for i in out_layers_ids]
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
