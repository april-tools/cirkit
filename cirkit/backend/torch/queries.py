import functools
from abc import ABC

from torch import Tensor
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchLayer, TorchInputLayer
from cirkit.utils.scope import Scope


class Query(ABC):
    def __init__(self) -> None:
        ...


class IntegrateQuery(Query):
    def __init__(self, circuit: TorchCircuit) -> None:
        super().__init__()
        self._circuit = circuit

    def __call__(self, x: Tensor, *, mar: Scope) -> Tensor:
        if not mar <= self._circuit.scope:
            raise ValueError("The variables to marginalize must be a subset of the circuit scope")
        return self._circuit.evaluate(
            x, module_fn=functools.partial(IntegrateQuery._layer_fn, mar=mar)
        )

    @staticmethod
    def _layer_fn(layer: TorchLayer, x: Tensor, mar: Scope) -> Tensor:
        if not isinstance(layer, TorchInputLayer) or len(layer.scope & mar) == 0:
            return layer(x)
        if len(layer.scope) > 1:
            raise NotImplementedError("Integration of multivariate input layers is not supported")
        output = layer.integrate()  # (F, 1, Ko)
        batch_size = x.shape[2]
        return output.expand(output.shape[0], batch_size, output.shape[2])
