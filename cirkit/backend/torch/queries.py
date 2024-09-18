import functools
from abc import ABC

import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.utils.scope import Scope


class Query(ABC):
    def __init__(self) -> None:
        ...


class IntegrateQuery(Query):
    def __init__(self, circuit: TorchCircuit) -> None:
        super().__init__()
        self._circuit = circuit

    def __call__(self, x: Tensor, *, vs: Scope) -> Tensor:
        if not vs <= self._circuit.scope:
            raise ValueError("The variables to marginalize must be a subset of the circuit scope")
        vs_idx = torch.tensor(tuple(vs))
        output = self._circuit.evaluate(
            x, module_fn=functools.partial(IntegrateQuery._layer_fn, vs_idx=vs_idx)
        )  # (O, B, K)
        return output.transpose(0, 1)  # (B, O, K)

    @staticmethod
    def _layer_fn(layer: TorchLayer, x: Tensor, vs_idx: Tensor) -> Tensor:
        output = layer(x)  # (F, B, Ko)
        if not isinstance(layer, TorchInputLayer):
            return output
        if layer.num_variables > 1:
            raise NotImplementedError("Integration of multivariate input layers is not supported")
        integration_mask = torch.isin(layer.scope_idx, vs_idx)  # Boolean mask of shape (F, 1)
        should_integrate = torch.any(integration_mask)
        if not should_integrate:
            return output
        # output: output of the layer of shape (F, B, Ko)
        # integration_mask: Boolean mask of shape (F, 1, 1)
        # integration_output: result of the integration of the layer of shape (F, 1, Ko)
        integration_mask = integration_mask.unsqueeze(dim=2)
        integration_output = layer.integrate()
        # Use the integration mask to select which output should be the result of
        # an integration operation, and which should not be
        # This is done in parallel for all folds, and regardless of whether the
        # circuit is folded or unfolded
        return torch.where(integration_mask, integration_output, output)
