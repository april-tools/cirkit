from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
from cirkit.backend.torch.layers import (
    TorchHadamardLayer,
    TorchInputLayer,
    TorchLayer,
    TorchSumLayer,
)
from cirkit.backend.torch.utils import CachedGateFunctionEval
from cirkit.symbolic.circuit import CircuitOperation, StructuralProperties
from cirkit.utils.conditional import GateFunctionParameterSpecs
from cirkit.utils.scope import Scope


class LayerAddressBook(AddressBook[TorchLayer]):
    """The address book data structure for the circuits.
    See [TorchCircuit][cirkit.backend.torch.circuits.TorchCircuit].
    The address book stores a list of
    [AddressBookEntry][cirkit.backend.torch.graph.modules.AddressBookEntry],
    where each entry stores the information needed to gather the inputs to each (possibly folded)
    circuit layer.
    """

    def backtrack(self, state, module_idxs: list[Tensor]) -> Tensor:
        # entry queue holds a tuple of the form:
        # (module address book id, idxs of module, previous module address book id)
        entry_queue = deque([(len(self._entries) - 1, None, None, None)])
        while entry_queue:
            entry_id, p_fold_idx, p_batch_idx, p_unit_idx = entry_queue.popleft()

            entry = self._entries[entry_id]
            module = entry.module
            in_module_ids = entry.in_module_ids

            if in_module_ids:
                in_modules_ids_h = in_module_ids[0]

                if module is None:
                    entry_queue.extend(
                        [
                            (
                                next_m_id,
                                0,
                                torch.arange(state.size(0), device=state.device),
                                0,
                            )
                            for next_m_id in in_modules_ids_h
                        ]
                    )
                    continue

                match module:
                    case TorchSumLayer():
                        in_fold_info = self._fold_idx_info.in_fold_idx[entry_id][p_fold_idx]

                        # retrieve arity and unit indexes by unraveling each batch
                        raveled_idxs = module_idxs[entry_id][p_fold_idx, p_batch_idx, p_unit_idx]
                        arity_idxs, unit_idxs = torch.unravel_index(
                            raveled_idxs, (module.arity, module.num_input_units)
                        )

                        for arity_i, (next_module_id, fold_idx) in enumerate(in_fold_info):
                            batch_idxs_at_arity = arity_idxs == arity_i

                            if batch_idxs_at_arity.any():
                                # specify which states are interested in the input module i
                                # and for each state which unit they are interested in
                                entry_queue.append(
                                    (
                                        next_module_id,
                                        fold_idx,
                                        p_batch_idx[batch_idxs_at_arity],
                                        unit_idxs[batch_idxs_at_arity],
                                    )
                                )

                        continue
                    case _:
                        in_fold_info = self._fold_idx_info.in_fold_idx[entry_id][p_fold_idx]
                        # for product layers we visit all the children
                        for next_module_id, fold_idx in in_fold_info:
                            entry_queue.append((next_module_id, fold_idx, p_batch_idx, p_unit_idx))
                        continue
            # catch the case where we are at an input unit
            elif module is not None:
                assert isinstance(module, TorchInputLayer)
                # set state
                input_idxs = module_idxs[entry_id]
                
                # check that the module is not a marginalized input
                # in that case ignore the update of this element
                if module.scope_idx.nelement() > 0:
                    state[p_batch_idx, module.scope_idx[p_fold_idx]] = input_idxs[
                        p_fold_idx, 0 if input_idxs.size(1) == 1 else p_batch_idx, p_unit_idx
                    ]

        return state

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
                    # when parameters are batched inputs from constant layers
                    # might have a batch size of 1 if in_graph is always None
                    # we have expand those inputs to match the others
                    module_inputs = [module_outputs[mid] for mid in in_layer_ids_h]

                    # make sure that all inputs have the same batch size
                    # if they do not, then it must be possible to partition them
                    # into two group where one group can be broadcast to the other
                    # group shape
                    # TODO: check for a more efficient implementation than casting to a set
                    batch_sizes = sorted([i.size(1) for i in module_inputs])
                    unique_batch_sizes = len(set(batch_sizes))
                    if unique_batch_sizes == 2:
                        # broadcast inputs with singleton batch size
                        module_inputs = [
                            i if i.size(1) != 1 else i.expand(-1, batch_sizes[-1], -1)
                            for i in module_inputs
                        ]
                    elif unique_batch_sizes > 2:
                        raise ValueError("Found an inconsistent batch dimension between units.")

                    x = torch.cat(module_inputs, dim=0)
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

        return LayerAddressBook(entries, fold_idx_info=fold_idx_info)


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
        gate_function_evals: Mapping[Mapping[str, CachedGateFunctionEval]] | None = None,
        symbolic_operation: CircuitOperation | None = None,
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
            gate_function_evals: A mapping from external gate functions to cached evaluations.
            symbolic_operation: The symbolic operation that created the circuit, if any.
        """
        super().__init__(
            layers,
            in_layers,
            outputs,
            fold_idx_info=fold_idx_info,
        )
        self._scope = scope
        self._properties = properties
        gate_function_evals = {} if gate_function_evals is None else gate_function_evals
        self._gate_function_evals = gate_function_evals
        self._symbolic_operation = symbolic_operation

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
    def device(self) -> torch.device:
        """Retrieve the device on which the circuit is loaded.

        Returns:
            torch.device: The device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def symbolic_operation(self) -> CircuitOperation:
        """Retrieve the symbolic operation that created the circuit.

        Returns:
            The symbolic operation.
        """
        return self._symbolic_operation

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

    @property
    def gate_function_evals(self) -> Mapping[str, CachedGateFunctionEval]:
        """Return the mapping between a gate function and its evaluation.

        Returns:
            Mapping[str, CachedGateFunctionEval]: The mapping between a gate
                function and its evaluation.
        """
        return self._gate_function_evals

    def reset_parameters(self) -> None:
        """Reset the parameters of the circuit in-place."""
        # For each layer, initialize its parameters, if any
        for l in self.layers:
            for p in l.params.values():
                p.reset_parameters()

    def _build_unfold_index_info(self) -> FoldIndexInfo:
        return build_unfold_index_info(
            self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
        )

    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> LayerAddressBook:
        return LayerAddressBook.from_index_info(fold_idx_info, incomings_fn=self.layer_inputs)

    def _memoize_gate_functions(self, gate_function_kwargs: Mapping[str, Mapping[str, Any]]):
        for gate_function_id, gate_function_eval in self._gate_function_evals.items():
            kwargs = gate_function_kwargs.get(gate_function_id, {})
            # memoize the gate function execution
            gate_function_eval.memoize(**kwargs)

    def __call__(
        self, x: Tensor | None = None, gate_function_kwargs: Mapping[str, Mapping[str, Any]] | None = None
    ) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x, gate_function_kwargs=gate_function_kwargs)  # type: ignore[no-any-return,misc]

    def forward(
        self, x: Tensor | None = None, gate_function_kwargs: Mapping[str, Mapping[str, Any]] | None = None) -> Tensor:
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
        return self._evaluate_layers(x, gate_function_kwargs=gate_function_kwargs)

    def _evaluate_layers(self, x: Tensor | None, gate_function_kwargs: Mapping[str, Mapping[str, Any]] | None = None) -> Tensor:
        # Memoize the gate functions.This will be called just before the invocation of the
        # [evaluate][cirkit.backend.torch.graph.modules.TorchDiAcyclicGraph.evaluate] method.
        self._memoize_gate_functions({} if gate_function_kwargs is None else gate_function_kwargs)
        
        # Evaluate layers on the given input
        y = self.evaluate(x)  # (O, B, K)
        y = y.transpose(0, 1)  # (B, O, K)
        # If the circuit has empty scope, we squeeze the batch dimension, as it is 1
        if not self._scope:
            y = y.squeeze(dim=0)  # (O, K)
        return y
