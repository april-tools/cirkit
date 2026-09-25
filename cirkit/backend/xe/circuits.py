from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.backend.torch.parameters.nodes import TorchParameterInput
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.xe.layers import XEInputLayer, XELayer, XESumLayer
from cirkit.backend.xe.program import (
    XECompilationConfig,
    XEProgramExecutor,
    build_executor,
)
from cirkit.symbolic.circuit import StructuralProperties
from cirkit.utils.scope import Scope


class XETorchCircuit(nn.Module):  # pylint: disable=too-many-instance-attributes
    """The extended-einsum circuit implementation, executed with torch.

    Evaluation combines a folded Cirkit torch frontend with an extended-einsum
    program. Input layers are evaluated by their compiled torch modules in linear
    space. Sum-layer parameter graphs that have an XE lowering supply their raw
    folded parameter tensors to the program, which evaluates the lowered parameter
    operations, all other parameter graphs are evaluated by their folded torch
    modules and supply their computed tensors instead. The program evaluates those
    inputs together with the circuit's inner layers.

    Programs are lowered lazily for each encountered batch-size and device pair,
    since the extended-einsum intermediate representation is shape-specialized.
    All frontend and program operations remain torch operations and can therefore
    be captured together by an outer ``torch.compile`` call.
    """

    def __init__(
        self,
        scope: Scope,
        layers: Sequence[XELayer],
        in_layers: Sequence[Sequence[int]],
        outputs: Sequence[int],
        *,
        properties: StructuralProperties,
        config: XECompilationConfig,
    ) -> None:
        """Initializes an extended-einsum circuit.

        Args:
            scope: The variables scope.
            layers: The compiled layer representations, in topological order.
            in_layers: For each layer, the indices of the layers that are input to it.
            outputs: The indices of the output layers.
            properties: The structural properties of the circuit.
            config: The compilation configuration.
        """
        super().__init__()
        self._scope = scope
        self._properties = properties
        self._config = config
        self._layers = list(layers)
        self._in_layers = [list(in_idx) for in_idx in in_layers]
        self._outputs = list(outputs)
        # Register the folded torch modules that own and evaluate input layers.
        # Their order matches the leaf-source indices used by the XE executor.
        self._input_modules = nn.ModuleList(
            dict.fromkeys(xl.module for xl in self._layers if isinstance(xl, XEInputLayer))
        )

        # Always register each complete sum-weight parameter graph: it owns the
        # trainable tensors, preserves sharing, supports parameter resetting, and is
        # the evaluator used when the graph has no XE lowering. The order matches the
        # weight-source indices used by the XE executor.
        self._weight_parameter_graphs = nn.ModuleList(
            dict.fromkeys(xl.weight for xl in self._layers if isinstance(xl, XESumLayer))
        )

        # Select once what each registered graph supplies to the XE executor. A
        # source is the raw parameter input when the remaining graph operations are
        # lowered into XE, and otherwise the complete graph itself. The raw inputs
        # remain owned and registered by their corresponding complete graphs.
        raw_weight_input_by_graph: dict[TorchParameter, TorchParameterInput] = {
            xl.weight: xl.weight_input
            for xl in self._layers
            if isinstance(xl, XESumLayer) and xl.weight_input is not None
        }
        weight_sources: list[TorchParameter | TorchParameterInput] = []
        for graph in self._weight_parameter_graphs:
            parameter_graph = cast(TorchParameter, graph)
            raw_input = raw_weight_input_by_graph.get(parameter_graph)
            weight_sources.append(raw_input if raw_input is not None else parameter_graph)
        self._weight_sources: tuple[TorchParameter | TorchParameterInput, ...] = tuple(
            weight_sources
        )
        # The lowered programs, one for each (batch size, device) encountered
        self._executors: dict[tuple[int, str], XEProgramExecutor] = {}

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
        return len(self._scope)

    @property
    def properties(self) -> StructuralProperties:
        """Retrieve the structural properties of the circuit.

        Returns:
            The structural properties.
        """
        return self._properties

    @property
    def layers(self) -> Sequence[XELayer]:
        """Retrieve the compiled layer representations.

        Returns:
            The compiled layers, in topological order.
        """
        return self._layers

    @property
    def semiring(self) -> str:
        """Retrieve the name of the semiring defining the circuit output space.

        Returns:
            The semiring name, either 'lse-sum' or 'sum-product'.
        """
        return self._config.semiring

    def reset_parameters(self) -> None:
        """Reset the parameters of the circuit in-place."""

        def _reset_input_layer(layer: TorchLayer) -> None:
            for p in layer.params.values():
                p.reset_parameters()
            for sub_layer in layer.sub_modules.values():
                _reset_input_layer(sub_layer)

        for module in self._input_modules:
            _reset_input_layer(cast(TorchInputLayer, module))
        for graph in self._weight_parameter_graphs:
            cast(TorchParameter, graph).reset_parameters()

    def __call__(self, x: Tensor | None = None) -> Tensor:
        return super().__call__(x)

    def forward(self, x: Tensor | None = None) -> Tensor:
        """Evaluate the circuit.

        Args:
            x: The tensor input of the circuit, with shape $(B, D)$, where B is the
                batch size, and $D$ is the number of variables. It can be None if the
                circuit has empty scope, i.e., it computes a constant tensor.
                Defaults to None.

        Returns:
            Tensor: The tensor output of the circuit, with shape $(B, O, K)$,
                where $O$ is the number of vectorized outputs (i.e., the number of
                output layers), and $K$ is the number of scalars in each output.
                If the circuit has empty scope and no input is given, then the
                output has shape $(O, K)$.

        Raises:
            ValueError: If the scope is not empty and the input to the circuit is
                None, or if the given input does not have two dimensions.
        """
        if self._scope and x is None:
            raise ValueError(f"Expected some input 'x', as the circuit has scope '{self._scope}'")
        if x is not None and len(x.shape) != 2:
            raise ValueError(
                "The input to the circuit should have shape (B, D), "
                "where B is the batch size and D is the number of variables "
                "the circuit is defined on"
            )
        batch_size = 1 if x is None else x.shape[0]
        device = x.device if x is not None else self._device()

        # Evaluate the input layers in linear space:
        # each output is a tensor of shape (B, K)
        folded_leaves: list[Tensor] = []
        for module in self._input_modules:
            assert isinstance(module, TorchInputLayer)
            if module.num_variables:
                assert x is not None
                # x: (B, D) -> (F, B, D')
                xm = x[..., module.scope_idx].permute(1, 0, 2)
                folded_leaves.append(module(xm))
            else:
                folded_leaves.append(module(batch_size))

        # Produce the sum-weight sources expected by the XE executor. A source is
        # either a raw folded parameter tensor whose remaining operations run in XE,
        # or the result of evaluating the complete folded torch parameter graph.
        weight_sources: list[Tensor] = [source() for source in self._weight_sources]

        # Execute the extended-einsum program lowered for this batch size and device
        executor = self._executor(batch_size, device)
        y = executor(folded_leaves, weight_sources)

        # y: (B, *U) or (O, B, *U) -> (B, O, K)
        if executor.num_outputs == 1:
            y = y.reshape(batch_size, 1, executor.num_output_units)
        else:
            y = y.reshape(executor.num_outputs, batch_size, executor.num_output_units)
            y = y.transpose(0, 1)
        # If the circuit has empty scope, we squeeze the batch dimension, as it is 1
        if not self._scope:
            y = y.squeeze(dim=0)
        return y

    def _device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        if parameter is None:
            return torch.device("cpu")
        return parameter.device

    def _executor(self, batch_size: int, device: torch.device) -> XEProgramExecutor:
        key = batch_size, str(device)
        executor = self._executors.get(key)
        if executor is None:
            executor = build_executor(
                self._layers,
                self._in_layers,
                self._outputs,
                batch_size=batch_size,
                device=device,
                config=self._config,
            )
            self._executors[key] = executor
        return executor
