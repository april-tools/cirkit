from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.graph.modules import (
    AddressBook,
    FoldIndexInfo,
    TorchDiAcyclicGraph,
    build_fold_index_info,
)
from cirkit.backend.torch.layers import TorchLayer
from cirkit.utils.scope import Scope


class LayerAddressBook(AddressBook):
    def lookup_module_inputs(
        self, module_id: int, module_outputs: List[Tensor], *, in_network: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        # Retrieve the input tensors given by other modules
        entry = self._entries[module_id]
        (in_module_ids,) = entry.in_module_ids
        (in_fold_idx,) = entry.in_fold_idx

        # Catch the case there are no inputs coming from other modules
        if not in_module_ids:
            assert in_fold_idx is not None
            assert in_network is not None
            x = self._in_network_fn(in_network, in_fold_idx)
            return (x,)

        # Catch the case there are some inputs coming from other modules
        in_tensors = tuple(module_outputs[mid] for mid in in_module_ids)
        x = torch.cat(in_tensors, dim=0)
        x = x.unsqueeze(dim=0) if in_fold_idx is None else x[in_fold_idx]
        return (x,)

    def lookup_output(self, module_outputs: List[Tensor]) -> Tensor:
        (output,) = self.lookup_module_inputs(-1, module_outputs=module_outputs)
        return output

    @classmethod
    def from_index_info(cls, fold_idx_info: FoldIndexInfo) -> "LayerAddressBook":
        ...


class AbstractTorchCircuit(TorchDiAcyclicGraph[TorchLayer]):
    def __init__(
        self,
        scope: Scope,
        num_channels: int,
        layers: List[TorchLayer],
        in_layers: Dict[TorchLayer, List[TorchLayer]],
        out_layers: Dict[TorchLayer, List[TorchLayer]],
        *,
        topologically_ordered: bool = False,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ) -> None:
        super().__init__(layers, in_layers, out_layers, topologically_ordered=topologically_ordered)
        self.scope = scope
        self.num_channels = num_channels
        self._fold_idx_info = fold_idx_info

    def reset_parameters(self) -> None:
        # For each layer, initialize its parameters, if any
        for l in self.layers:
            for _, p in l.params.items():
                if not p.has_address_book:
                    p.initialize_address_book()
                p.initialize_()

    def layer_inputs(self, l: TorchLayer) -> List[TorchLayer]:
        return self.node_inputs(l)

    def layer_outputs(self, l: TorchLayer) -> List[TorchLayer]:
        return self.node_outputs(l)

    @property
    def layers(self) -> List[TorchLayer]:
        return self.nodes

    @property
    def layers_inputs(self) -> Dict[TorchLayer, List[TorchLayer]]:
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Dict[TorchLayer, List[TorchLayer]]:
        return self.nodes_outputs

    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        return LayerAddressBook.from_index_info(fold_idx_info)

    def _in_index(self, x: Tensor, idx: Tensor) -> Tensor:
        # Index and process the input tensor, before feeding it to the input layers
        # x: (B, C, D)
        x = x[..., idx]  # (B, C, F, D)
        x = x.permute(2, 1, 0, 3)  # (F, C, B, D)
        return x

    def _eval_layers(self, x: Tensor) -> Tensor:
        # Evaluate layers
        y = self._eval_forward(x)  # (1, num_classes, B, K)
        y = y.squeeze(dim=0)  # (num_classes, B, K)
        y = y.transpose(0, 1)  # (B, num_classes, K)
        return y


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
        return self._eval_layers(x)


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
        x = self._eval_layers(x)
        return x.squeeze(dim=0)  # squeeze dummy batch dimension
