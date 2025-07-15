from collections.abc import Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from cirkit.backend.torch.graph.folding import (
    build_address_book_stacked_entry,
    build_unfold_index_info,
)
from cirkit.backend.torch.graph.modules import (
    AddressBook,
    AddressBookEntry,
    FoldIndexInfo,
    TorchDiAcyclicGraph,
)
from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.symbolic.circuit import StructuralProperties
from cirkit.utils.scope import Scope


class LayerAddressBook(AddressBook[TorchLayer]):
    """The address book data structure for the circuits.
    See [TorchCircuit][cirkit.backend.torch.circuits.TorchCircuit].
    The address book stores a list of
    [AddressBookEntry][cirkit.backend.torch.graph.modules.AddressBookEntry],
    where each entry stores the information needed to gather the inputs to each (possibly folded)
    circuit layer.
    """

    def lookup(
        self, module_outputs: list[Tensor], *, in_graph: Tensor | None = None
    ) -> Iterator[tuple[TorchLayer | None, tuple]]:
        # Loop through the entries and yield inputs
        for entry in self:
            layer = entry.module
            in_layer_ids = entry.in_module_ids
            in_fold_idx = entry.in_fold_idx
            # Catch the case there are some inputs coming from other modules
            if in_layer_ids:
                in_fold_idx_h = in_fold_idx[0]
                in_layer_ids_h = in_layer_ids[0]
                if len(in_layer_ids_h) == 1:
                    x = module_outputs[in_layer_ids_h[0]]
                else:
                    x = torch.cat([module_outputs[mid] for mid in in_layer_ids_h], dim=0)
                x = x[in_fold_idx_h]
                yield layer, (x,)
                continue

            # Catch the case there are no inputs coming from other modules
            # That is, we are gathering the inputs of input layers
            assert isinstance(layer, TorchInputLayer)
            if layer.num_variables:
                if in_graph is None:
                    yield layer, ()
                    continue
                # in_graph: An input batch (assignments to variables) of shape (B, D)
                # scope_idx: The scope of the layers in each fold, a tensor of shape (F, D'), D' < D
                # x: (B, D) -> (B, F, D') -> (F, B, D')
                if len(in_graph.shape) != 2:
                    raise ValueError(
                        "The input to the circuit should have shape (B, D), "
                        "where B is the batch size and D is the number of variables "
                        "the circuit is defined on"
                    )
                x = in_graph[..., layer.scope_idx].permute(1, 0, 2)
                yield layer, (x,)
                continue

            # Pass the wanted batch dimension to constant layers
            yield layer, (1 if in_graph is None else in_graph.shape[0],)

    @classmethod
    def from_index_info(
        cls,
        fold_idx_info: FoldIndexInfo[TorchLayer],
        *,
        incomings_fn: Callable[[TorchLayer], Sequence[TorchLayer]],
    ) -> "LayerAddressBook":
        """Constructs the layers address book using fold index information.

        Args:
            fold_idx_info: The fold index information.
            incomings_fn: A function mapping each circuit layer to the sequence of its inputs.

        Returns:
            A layers address book.
        """
        # The address book entries being built
        entries: list[AddressBookEntry[TorchLayer]] = []

        # A useful dictionary mapping module ids to their number of folds
        num_folds: dict[int, int] = {}

        # Build the bookkeeping data structure by following the topological ordering
        for mid, m in enumerate(fold_idx_info.ordering):
            # Retrieve the index information of the input modules
            in_modules_fold_idx = fold_idx_info.in_fold_idx[mid]

            # Catch the case of a folded module having the output of another module as input
            if incomings_fn(m):
                entry = build_address_book_stacked_entry(
                    m, in_modules_fold_idx, num_folds=num_folds
                )
            else:
                # Catch the case of a folded module having the input of the network as input
                # That is, this is the case of an input layer
                entry = AddressBookEntry(m, [], [])

            num_folds[mid] = m.num_folds
            entries.append(entry)

        # Append the last bookkeeping entry with the information to compute the output tensor
        entry = build_address_book_stacked_entry(
            None, [fold_idx_info.out_fold_idx], num_folds=num_folds, output=True
        )
        entries.append(entry)

        return LayerAddressBook(entries)


class TorchCircuit(TorchDiAcyclicGraph[TorchLayer]):
    """The torch circuit implementation. It is a (possibly folded)
    computational graph of torch layers implementations."""

    def __init__(
        self,
        scope: Scope,
        layers: Sequence[TorchLayer],
        in_layers: Mapping[TorchLayer, Sequence[TorchLayer]],
        outputs: Sequence[TorchLayer],
        *,
        properties: StructuralProperties,
        fold_idx_info: FoldIndexInfo[TorchLayer] | None = None,
    ) -> None:
        """Initializes a torch circuit.

        Args:
            scope: The variables scope.
            layers: The sequence of layers.
            in_layers: A dictionary mapping layers to their inputs, if any.
            outputs: A list of output layers.
            properties: The structural properties of the circuit.
            fold_idx_info: The folding index information.
                It can be None if the circuit is not folded.
        """
        super().__init__(
            layers,
            in_layers,
            outputs,
            fold_idx_info=fold_idx_info,
        )
        self._scope = scope
        self._properties = properties

    @property
    def scope(self) -> Scope:
        """Retrieve the variables scope of the circuit.

        Returns:
            The scope.
        """
        return self._scope

    @property
    def num_variables(self) -> int:
        """Retrieve the number of variables the circuit is defined on.

        Returns:
            The number of variables.
        """
        return len(self.scope)

    @property
    def properties(self) -> StructuralProperties:
        """Retrieve the structural properties of the circuit.

        Returns:
            The structural properties.
        """
        return self._properties

    @property
    def layers(self) -> Sequence[TorchLayer]:
        """Retrieve the layers.

        Returns:
            The layers.
        """
        return self.nodes

    def layer_inputs(self, l: TorchLayer) -> Sequence[TorchLayer]:
        """Given a layer, retrieve the layers that are input to it.

        Args:
            l: The layer.

        Returns:
            The inputs to the given layer.
        """
        return self.node_inputs(l)

    def layer_outputs(self, l: TorchLayer) -> Sequence[TorchLayer]:
        """Given a layer, retrieve the layers that receive input from it.

        Args:
            l: The layer.

        Returns:
            The outputs from the given layer.
        """
        return self.node_outputs(l)

    @property
    def layers_inputs(self) -> Mapping[TorchLayer, Sequence[TorchLayer]]:
        """Retrieve the map from layers to their inputs.

        Returns:
            The layers inputs map.
        """
        return self.nodes_inputs

    @property
    def layers_outputs(self) -> Mapping[TorchLayer, Sequence[TorchLayer]]:
        """Retrieve the map from layers to their outputs.

        Returns:
            The layers outputs map.
        """
        return self.nodes_outputs

    def reset_parameters(self) -> None:
        """Reset the parameters of the circuit in-place."""
        # For each layer, initialize its parameters, if any
        for l in self.layers:
            for p in l.params.values():
                p.reset_parameters()

    def __call__(self, x: Tensor | None = None) -> Tensor:
        return super().__call__(x)

    def forward(self, x: Tensor | None = None) -> Tensor:
        """Evaluate the circuit layers in forward mode, i.e., by evaluating each layer by
        following the topological ordering.

        Args:
            x: The tensor input of the circuit, with shape $(B, D)$, where B is the batch size,
                and $D$ is the number of variables. It can be None if the circuit has empty scope,
                i.e., it computes a constant tensor. Defaults to None.

        Returns:
            Tensor: The tensor output of the circuit, with shape $(B, O, K)$,
                where $O$ is the number of vectorized outputs (i.e., the number of output layers),
                and $K$ is the number of scalars in each output (e.g., the number of classes).

        Raises:
            ValueError: If the scope is not empty and the tensor input to the circuit is None.
        """
        if self._scope and x is None:
            raise ValueError(f"Expected some input 'x', as the circuit has scope '{self._scope}'")
        return self._evaluate_layers(x)

    def _build_unfold_index_info(self) -> FoldIndexInfo[TorchLayer]:
        return build_unfold_index_info(
            self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
        )

    def _build_address_book(self, fold_idx_info: FoldIndexInfo[TorchLayer]) -> LayerAddressBook:
        return LayerAddressBook.from_index_info(fold_idx_info, incomings_fn=self.layer_inputs)

    def _evaluate_layers(self, x: Tensor | None) -> Tensor:
        # Evaluate layers on the given input
        y = self.evaluate(x)  # (O, B, K)
        y = y.transpose(0, 1)  # (B, O, K)
        # If the circuit has empty scope, we squeeze the batch dimension, as it is 1
        if not self._scope:
            y = y.squeeze(dim=0)  # (O, K)
        return y
