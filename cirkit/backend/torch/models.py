from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph import TorchDiAcyclicGraph, AddressBook, AbstractAddressBook
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
        address_book: Optional[AbstractAddressBook] = None
    ) -> None:
        super().__init__()
        if address_book is None:
            address_book = AddressBook(
                in_address_fn=lambda l: list(l.scope),
                in_process_fn=lambda x: x.permute(2, 1, 0, 3),
                stack_in_tensors=True
            )
        self._graph = TorchDiAcyclicGraph[TorchLayer](
            layers, in_layers, out_layers,
            topologically_ordered=topologically_ordered,
            address_book=address_book,
        )
        self.scope = scope
        self.num_channels = num_channels

    def _eval_forward(self, x: Tensor) -> Tensor:
        y = self._graph.eval_forward(x)  # (1, num_classes, B, K)
        y = y.squeeze(dim=0)             # (num_classes, B, K)
        y = y.transpose(0, 1)  # (B, num_classes, K)
        return y

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
