from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from cirkit.backend.torch.layers import (
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchLayer,
    TorchSumLayer,
    TorchTuckerLayer,
)
from cirkit.backend.torch.layers.optimized import TorchCPTLayer, TorchTensorDotLayer
from cirkit.backend.torch.optimization.parameters import KroneckerOutParameterPattern
from cirkit.backend.torch.optimization.registry import (
    LayerOptApplyFunc,
    LayerOptMatch,
    LayerOptPattern,
    LayerOptPatternDefn,
    ParameterOptPattern,
)
from cirkit.backend.torch.parameters.nodes import (
    TorchKroneckerParameter,
    TorchMatMulParameter,
)
from cirkit.backend.torch.parameters.parameter import TorchParameter

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


class SumCollapsePattern(LayerOptPatternDefn):
    """Detect adjacent sum layers that could be fused"""

    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> Sequence[type[TorchLayer]]:
        return [TorchSumLayer, TorchSumLayer]

    @classmethod
    def sub_patterns(cls) -> Sequence[dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]

    @classmethod
    def config_patterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}, {}]


class TuckerPattern(LayerOptPatternDefn):
    """Detect combinations of Sum and Kroenecker product to merge in a Tucker layer"""

    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> Sequence[type[TorchLayer]]:
        return [TorchSumLayer, TorchKroneckerLayer]

    @classmethod
    def sub_patterns(cls) -> Sequence[dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]

    @classmethod
    def config_patterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}, {}]


class CandecompPattern(LayerOptPatternDefn):
    """Detect combinations of Hadamard and Sum layer to merge as CP-T layers."""

    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> Sequence[type[TorchLayer]]:
        return [TorchSumLayer, TorchHadamardLayer]

    @classmethod
    def sub_patterns(cls) -> Sequence[dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]

    @classmethod
    def config_patterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}, {}]


class DenseKroneckerPattern(LayerOptPatternDefn):
    r"""Detect sum layer which have a Kronecker parameter node as parameter output.

    The goal of this pattern is to replace the expensive matrix multiplication from
    the sum by leveraging the decomposition of the parameters from the kronecker product.
    
    Given $W=A \otimes B$ the parameters of the sum layer,
    with $A$ of shape $(a_1,\dots,a_n)$ and $B$ of shape $(b_1,\dots,b_n)$
    $$
        \begin{align*} 
        (Wx)_{kl} &=((A \otimes B) x)_{kl}\\
        &= (B (A x)^{T})_{k1}  
        \end{align*}
    $$
    
    As $W$ has shape $(a_1b_1,\dots,\a_nb_n)$, it is significantly less computationally
    expensive to compute the two inner products instead.
    """

    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> Sequence[type[TorchLayer]]:
        return [TorchSumLayer]

    @classmethod
    def sub_patterns(cls) -> Sequence[dict[str, ParameterOptPattern]]:
        return [{"weight": KroneckerOutParameterPattern}]

    @classmethod
    def config_patterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}]


class TensorDotKroneckerPattern(LayerOptPatternDefn):
    r"""Detect Dot layer which have a Kronecker parameter node as parameter output.

    The goal of this pattern is to replace the expensive matrix multiplication from
    the dot layer by leveraging the decomposition of the parameters from the kronecker product.
    
    Given $W=A \otimes B$ the parameters of the dot layer,
    with $A$ of shape $(a_1,\dots,a_n)$ and $B$ of shape $(b_1,\dots,b_n)$
    $$
        \begin{align*} 
        (Wx)_{kl} &=((A \otimes B) x)_{kl}\\
        &= (B (A x)^{T})_{k1}  
        \end{align*}
    $$
    
    As $W$ has shape $(a_1b_1,\dots,\a_nb_n)$, it is significantly less computationally
    expensive to compute the two inner products instead.
    """

    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> Sequence[type[TorchLayer]]:
        return [TorchTensorDotLayer]

    @classmethod
    def sub_patterns(cls) -> Sequence[dict[str, ParameterOptPattern]]:
        return [{"weight": KroneckerOutParameterPattern}]

    @classmethod
    def config_patterns(cls) -> list[dict[str, Any]]:
        return [{}]


def apply_sum_collapse(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> tuple[TorchSumLayer]:
    """Fuse two sum nodes together.

    This function simply develop the two node into one
    single sum using matrix multiplication of the two
    sum's parameters.

    Indeed, if we have two sums with parameters $W_1$, $W_2$:
    $$S_1=W_1X$$
    $$S_2=W_2S_1$$
    $$S_2=W_2W_1X$$

    The final sums have weight : $W_2W_1$

    Args:
        compiler (TorchCompiler): The current compiler
        match: The match to replace

    Returns:
       tuple[TorchSumLayer]: The sum layer computing the two sum
            in one sum.
    """
    dense1 = cast(TorchSumLayer, match.entries[0])
    dense2 = cast(TorchSumLayer, match.entries[1])
    weight = TorchParameter.from_binary(
        TorchMatMulParameter(dense1.weight.shape, dense2.weight.shape),
        dense1.weight,
        dense2.weight,
    )
    dense = TorchSumLayer(
        dense2.num_input_units,
        dense1.num_output_units,
        arity=dense2.arity,
        weight=weight,
        semiring=compiler.semiring,
    )
    return (dense,)


def apply_tucker(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> tuple[TorchTuckerLayer]:
    r"""Create a Tucker layer that compute the sum of a kronecker product.

    This optimization consists of rewriting the full operation in a single
    einsum to avoid computing the intermediary tensor from the kronecker
    product.

    The output of the kronecker product which take the vectors $x$ and $y$ of shape $a$ and
    $b$ respectively (no batch or fold for simplicity), can be written as the
    following einsum:

    $$
        a,b \rightarrow ab
    $$

    We would then proceed to flatten the output to get a vector $z$ of
    size $i=a \times b$. This vector is then used in the einsum for the sum.
    Given W the parameter matrix of shape $(o,i)$, the sum $Wx$ is:

    $$
        i,oi \rightarrow o
    $$

    Now let's reshape the tensors to re-introduce the $a$ and $b$ dimensions.
    The sum would be written as:

    $$
        ab, oab \rightarrow o
    $$

    We can finally substitute the results of the kronecker product for the $x$
    and $y$ vectors:

    $$
        a,b,oab \rightarrow o
    $$

    Thus avoiding the intermediary Kronecker product.
    This is exactly what the tucker layer will compute.

    Args:
        compiler (TorchCompiler): The current compiler.
        match (LayerOptMatch): The match to replace.

    Returns:
       tuple[TorchTuckerLayer]: The tucker layer merging the two operations.
    """
    dense = cast(TorchSumLayer, match.entries[0])
    kronecker = cast(TorchKroneckerLayer, match.entries[1])
    tucker = TorchTuckerLayer(
        kronecker.num_input_units,
        dense.num_output_units,
        kronecker.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )
    return (tucker,)


def apply_candecomp(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> tuple[TorchCPTLayer]:
    r"""Construct the CPT layer fusing one Sum and one Hadamard layer.

    Args:
        compiler (TorchCompiler): The current compiler doing the optimization.
        match (LayerOptMatch): The match to optimize.

    Returns:
        tuple[TorchCPTLayer]: The CPT layer replacing the sum and hadamard layers.
    """
    dense = cast(TorchSumLayer, match.entries[0])
    hadamard = cast(TorchHadamardLayer, match.entries[1])
    cpt = TorchCPTLayer(
        hadamard.num_input_units,
        dense.num_output_units,
        hadamard.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )
    return (cpt,)


def _apply_tensordot_rule(
    compiler: "TorchCompiler",
    num_input_units: int,
    num_output_units: int,
    weight: TorchParameter,
    kronecker: TorchKroneckerParameter,
) -> tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    r"""Returns the two dot layer representing a sum parameterized by the output of a Kronecker Product.

    This trick comes from (Zhang et al., 2025) Subsection 3.1.
    Given $W=A \otimes B$ the parameters of a sum or dot layer,
    with $A$ of shape $(a_1,\dots,a_n)$ and $B$ of shape $(b_1,\dots,b_n)$
            $$
                \begin{align*} 
                (Wx)_{kl} &=((A \otimes B) x)_{kl}\\
                &= (B (A x)^{T})_{k1}  
                \end{align*}
            $$

    We can convince ourselves that it works by developing a simple example:
    First the normal kronecker and sum / dot:
        $$
            A=
            \begin{bmatrix}
            a_1 & a_2 \\ 
            a_3 & a_4
            \end{bmatrix}
            \text{ and }
            B=
            \begin{bmatrix}
            b_1 & b_2 \\ 
            b_3 & b_4
            \end{bmatrix}
        $$

    Then 
        $$
            W=A\otimes B= \begin{bmatrix} 
            a_1b_1 & a_1b_2 & a_2b_1 & a_2b_2 \\
            a_1b_3 & a_1b_4 & a_2b_3 & a_2b_4 \\
            a_3b_1 & a_3b_2 & a_4b_1 & a_4b_2 \\
            a_3b_3 & a_3b_4 & a_4b_3 & a_4b_4 \\
            \end{bmatrix}
        $$
    So the final sum would be :
        
        $$
            Wx=\begin{bmatrix} 
            a_1b_1 & a_1b_2 & a_2b_1 & a_2b_2 \\
            a_1b_3 & a_1b_4 & a_2b_3 & a_2b_4 \\
            a_3b_1 & a_3b_2 & a_4b_1 & a_4b_2 \\
            a_3b_3 & a_3b_4 & a_4b_3 & a_4b_4 \\
            \end{bmatrix}
            \begin{bmatrix}
            x_1\\x_2\\x_3\\x_4
            \end{bmatrix}
            =
            \begin{bmatrix}
            x_1a_1b_1 + x_2a_1b_2 + x_3a_2b_1 + x_4a_2b_2\\
            x_1a_1b_3 + x_2a_1b_4 + x_3a_2b_3 +x_4a_2b_4 \\
            \dots
            \end{bmatrix}
        $$
    Now let's compute $Ax$ following the dot layer procedures.
    Let's first reshape x:
        $$
            x=\begin{bmatrix} x_1 & x_2 \\ x_3 & x_4 \end{bmatrix}
        $$
    Now $Ax$ :
        $$
            \begin{bmatrix}
            a_1 & a_2 \\ 
            a_3 & a_4
            \end{bmatrix}
            \begin{bmatrix}
            x_1 & x_2 \\ 
            x_3 & x_4
            \end{bmatrix}
            =
            \begin{bmatrix}
            a_1x_1+a_2x_3 & a_1x_2+a_2x_4 \\ 
            a_3x_1+a_4x_3 & a_3x_2+a_4x_4
            \end{bmatrix}
        $$
    And $B(Ax)^T$ :
        $$
            \begin{bmatrix}
            b_1 & b_2 \\ 
            b_3 & b_4
            \end{bmatrix}
            \begin{bmatrix}
            a_1x_1+a_2x_3 & a_3x_1+a_4x_3  \\ 
            a_1x_2+a_2x_4 & a_3x_2+a_4x_4
            \end{bmatrix}
            =
            \begin{bmatrix}
            x_1a_1b_1 + x_2a_1b_2 + x_3a_2b_1 + x_4a_2b_2\\
            x_1a_1b_3 + x_2a_1b_4 + x_3a_2b_3 +x_4a_2b_4 \\
            \dots
            \end{bmatrix}
        $$

    We get the same results using both expression !

    Args:
        compiler (TorchCompiler): The current compiler.
        num_input_units (int): Number of input units on the sum layer.
        num_output_units (int): Number of output units on the sum layer.
        weight (TorchParameter): The weight graph with a Kronecker output.
        kronecker (TorchKroneckerParameter): The Kronecker parameter node.

    Returns:
        tuple[TorchTensorDotLayer, TorchTensorDotLayer]: the two dot layer to
            replace the original layer.

    References:
        Zhang, H., Dang, M., Wang, B., Ermon, S., Peng, N., & Broeck, G. V. den. (2025). 
        Scaling Probabilistic Circuits via Monarch Matrices (No. arXiv:2506.12383).
        arXiv. https://doi.org/10.48550/arXiv.2506.12383
    """
    # Build new torch parameter computational graphs by taking
    # the sub-computational graph rooted at the inputs of the kronecker parameter node
    in_kronecker1, in_kronecker2 = weight.node_inputs(kronecker)
    weight1 = weight.subgraph(in_kronecker1)
    weight2 = weight.subgraph(in_kronecker2)

    # Instantiate two tensor dot layers
    num_inner_units = weight1.shape[0] * (num_input_units // weight1.shape[1])
    tdot1 = TorchTensorDotLayer(
        num_input_units,
        num_inner_units,
        weight=weight1,
        semiring=compiler.semiring,
    )
    tdot2 = TorchTensorDotLayer(
        num_inner_units,
        num_output_units,
        weight=weight2,
        semiring=compiler.semiring,
    )
    return tdot1, tdot2


def apply_dense_tensordot(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    r"""Return two Dot Layer corresponding to a Sum parameterized by a Kronecker product

    Args:
        compiler (TorchCompiler): The current compiler doing the optimization.
        match (LayerOptMatch): The match to optimize.

    Returns:
        tuple[TorchTensorDotLayer, TorchTensorDotLayer]: the two dot layer to
            replace the sum layer.
    """
    dense = cast(TorchSumLayer, match.entries[0])
    weight_patterns = match.sub_entries[0]["weight"]
    kronecker = cast(TorchKroneckerParameter, weight_patterns[0].entries[0])
    return _apply_tensordot_rule(
        compiler, dense.num_input_units, dense.num_output_units, dense.weight, kronecker
    )


def apply_tensordot_tensordot(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    r"""Return two Dot Layer corresponding to a Dot Layer parameterized by a Kronecker product

    Args:
        compiler (TorchCompiler): The current compiler doing the optimization.
        match (LayerOptMatch): The match to optimize.

    Returns:
        tuple[TorchTensorDotLayer, TorchTensorDotLayer]: the two dot layer to
            replace the sum layer.
    """

    tdot = cast(TorchTensorDotLayer, match.entries[0])
    weight_patterns = match.sub_entries[0]["weight"]
    kronecker = cast(TorchKroneckerParameter, weight_patterns[0].entries[0])
    return _apply_tensordot_rule(
        compiler, tdot.num_input_units, tdot.num_output_units, tdot.weight, kronecker
    )


DEFAULT_LAYER_FUSE_OPT_RULES: Mapping[LayerOptPattern, LayerOptApplyFunc] = {
    SumCollapsePattern: apply_sum_collapse,
    TuckerPattern: apply_tucker,
    CandecompPattern: apply_candecomp,
}
DEFAULT_LAYER_SHATTER_OPT_RULES: Mapping[LayerOptPattern, LayerOptApplyFunc] = {
    DenseKroneckerPattern: apply_dense_tensordot,
    TensorDotKroneckerPattern: apply_tensordot_tensordot,
}
