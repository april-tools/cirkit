from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.graph.address_book import AddressBook, AddressBookEntry, FoldIndexInfo
from cirkit.backend.torch.graph.folding import (
    build_address_book_stacked_entry,
    build_fold_index_info,
)
from cirkit.backend.torch.graph.modules import TorchDiAcyclicGraph
from cirkit.backend.torch.layers import TorchLayer
from cirkit.utils.scope import Scope


class LayerAddressBook(AddressBook):
    def __init__(
        self, entries: List[AddressBookEntry], *, in_graph_fn: Callable[[Tensor, Tensor], Tensor]
    ):
        super().__init__(entries)
        self._in_graph_fn = in_graph_fn

    def lookup(
        self, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Iterator[Tuple[Tensor, ...]]:
        # Loop through the entries and yield inputs
        for entry in self._entries:
            (in_fold_idx,) = entry.in_fold_idx

            # Catch the case there are some inputs coming from other modules
            if entry.in_module_ids:
                (in_module_ids,) = entry.in_module_ids
                if len(in_module_ids) == 1:
                    x = module_outputs[in_module_ids[0]]
                else:
                    x = torch.cat([module_outputs[mid] for mid in in_module_ids], dim=0)
                x = x[in_fold_idx]
                yield (x,)
                continue

            # Catch the case there are no inputs coming from other modules
            assert in_graph is not None and in_fold_idx is not None
            x = self._in_graph_fn(in_graph, in_fold_idx)
            yield (x,)

    @classmethod
    def from_index_info(
        cls,
        ordering: Iterable[TorchLayer],
        fold_idx_info: FoldIndexInfo,
        *,
        incomings_fn: Callable[[TorchLayer], List[TorchLayer]],
        in_graph_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> "LayerAddressBook":
        # The address book entries being built
        entries: List[AddressBookEntry] = []

        # A useful dictionary mapping module ids to their number of folds
        num_folds: Dict[int, int] = {}

        # Build the bookkeeping data structure by following the topological ordering
        for mid, m in enumerate(ordering):
            # Retrieve the index information of the input modules
            in_modules_fold_idx = fold_idx_info.in_fold_idx[mid]

            # Catch the case of a folded module having the output of another module as input
            if incomings_fn(m):
                entry = build_address_book_stacked_entry(in_modules_fold_idx, num_folds=num_folds)
            # Catch the case of a folded module having the input of the network as input
            else:
                input_idx = [[idx[1] for idx in fi] for fi in in_modules_fold_idx]
                input_idx_t = torch.tensor(input_idx)
                entry = AddressBookEntry([], [input_idx_t])

            num_folds[mid] = m.num_folds
            entries.append(entry)

        # Append the last bookkeeping entry with the information to compute the output tensor
        entry = build_address_book_stacked_entry(
            [fold_idx_info.out_fold_idx], num_folds=num_folds, output=True
        )
        entries.append(entry)

        return LayerAddressBook(entries, in_graph_fn=in_graph_fn)


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
        super().__init__(
            layers,
            in_layers,
            out_layers,
            topologically_ordered=topologically_ordered,
            fold_idx_info=fold_idx_info,
        )
        self.scope = scope
        self.num_channels = num_channels

    def reset_parameters(self) -> None:
        # For each layer, initialize its parameters, if any
        for l in self.layers:
            for _, p in l.params.items():
                if not p.has_address_book:
                    p.initialize_address_book()
                p.reset_parameters()

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

    def _build_address_book(self) -> AddressBook:
        fold_idx_info = self._fold_idx_info
        if fold_idx_info is None:
            fold_idx_info = build_fold_index_info(
                self.topological_ordering(),
                outputs=self.outputs,
                incomings_fn=self.node_inputs,
                in_address_fn=lambda l: l.scope,
            )
        address_book = LayerAddressBook.from_index_info(
            self.topological_ordering(),
            fold_idx_info,
            incomings_fn=self.layer_inputs,
            in_graph_fn=self._index_input,
        )
        self._fold_idx_info = None
        return address_book

    def _index_input(self, x: Tensor, idx: Tensor) -> Tensor:
        # Index and process the input tensor, before feeding it to the input layers
        # x: (B, C, D)
        return x[..., idx].permute(2, 1, 0, 3)  # (F, C, B, D)

    def _eval_layers(self, x: Tensor) -> Tensor:
        # Evaluate layers
        y = self._eval_forward(x)  # (O, B, K)
        return y.transpose(0, 1)  # (B, O, K)


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
        x = torch.empty(size=(1, self.num_channels, len(self.scope)), device=self.device)
        x = self._eval_layers(x)  # (B, O, K)
        return x.squeeze(dim=0)  # (O, K)
