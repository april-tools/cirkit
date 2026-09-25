from abc import ABC
from dataclasses import dataclass

from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.backend.torch.parameters.nodes import TorchParameterInput
from cirkit.backend.torch.parameters.parameter import TorchParameter


class XELayer(ABC):
    """The abstract compiled layer representation used by the extended-einsum backend.

    A compiled XE layer does not perform any computation by itself. Instead, it stores
    the information needed to (i) evaluate the layer inputs and parameters with torch
    modules, and (ii) lower the layer computation into an extended-einsum program.
    See [XETorchCircuit][cirkit.backend.xe.circuits.XETorchCircuit] for more details.
    """


@dataclass(frozen=True, eq=False)
class XEInputLayer(XELayer):
    """A compiled input layer. The evaluation of an input layer is not part of the
    extended-einsum program. Instead, the compiled torch input layer is evaluated
    (in the sum-product semiring, i.e., in linear space) and its output enters the
    program as a data input of shape $(B, K)$, where $B$ is the batch size and $K$
    is the number of output units."""

    module: TorchInputLayer
    """The torch module evaluating the input layer in linear space."""
    num_output_units: int
    """The number of output units of the layer."""
    fold_idx: int = 0
    """The fold of the torch input module evaluating this layer."""


@dataclass(frozen=True, eq=False)
class XESumLayer(XELayer):
    """A compiled sum layer. The weight is computed by a torch parameter computational
    graph and enters the extended-einsum program as a parameter input, i.e., an input
    that the numerical stability translations keep in linear space. The weighted sum
    itself is encoded as an einsum operation in the program."""

    weight: TorchParameter
    """The torch parameter computational graph computing the weight of shape
    $(K_o, H \\cdot K_i)$."""
    num_input_units: int
    """The number of units in each layer that is input to this layer."""
    num_output_units: int
    """The number of sum units in the layer."""
    arity: int
    """The number of layers that are input to this layer."""
    fold_idx: int = 0
    """The fold of the torch parameter graph computing this layer's weight."""
    weight_input: TorchParameterInput | None = None
    """The raw folded parameter input when its activation is lowered into XE."""
    weight_softmax_dim: int | None = None
    """The softmax axis lowered into XE, excluding the fold dimension."""


@dataclass(frozen=True, eq=False)
class XEHadamardLayer(XELayer):
    """A compiled Hadamard product layer, encoded as an element-wise einsum operation
    in the extended-einsum program."""

    num_input_units: int
    """The number of units in each layer that is input to this layer."""
    arity: int
    """The number of layers that are input to this layer."""


@dataclass(frozen=True, eq=False)
class XEKroneckerLayer(XELayer):
    """A compiled Kronecker product layer, encoded as an outer-product einsum operation
    in the extended-einsum program. The units of the output are kept factorized as
    multiple tensor axes, i.e., the output of the einsum operation has shape
    $(B, K_1, \\ldots, K_H)$ rather than $(B, \\prod_i K_i)$, since the extended-einsum
    language does not have a reshape operation. Layers consuming a Kronecker layer
    output take this factorization into account, e.g., the weight of a sum layer is
    reshaped (outside of the program) to match the factorized axes."""

    num_input_units: int
    """The number of units in each layer that is input to this layer."""
    arity: int
    """The number of layers that are input to this layer."""
