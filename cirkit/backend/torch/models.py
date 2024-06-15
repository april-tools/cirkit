from typing import Dict, List, Optional

import torch
from torch import Tensor

from cirkit.backend.torch.graph.modules import TorchDiAcyclicGraph
from cirkit.backend.torch.graph.books import AbstractAddressBook, AddressBook
from cirkit.backend.torch.layers import TorchLayer
from cirkit.utils.scope import Scope


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
        address_book: Optional[AbstractAddressBook] = None
    ) -> None:
        if address_book is None:
            address_book = AddressBook(
                in_address_fn=lambda l: list(l.scope),
                stack_in_tensors=True
            )
        super().__init__(
            layers, in_layers, out_layers,
            topologically_ordered=topologically_ordered,
            address_book=address_book
        )
        self.scope = scope
        self.num_channels = num_channels

    def _in_index(self, x: Tensor, idx: Tensor) -> Tensor:
        x = super()._in_index(x, idx)
        return x.permute(2, 1, 0, 3)

    def _eval_layers(self, x: Tensor) -> Tensor:
        y = super()._eval_forward(x)      # (1, num_classes, B, K)
        y = y.squeeze(dim=0)              # (num_classes, B, K)
        y = y.transpose(0, 1)  # (B, num_classes, K)
        return y

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
