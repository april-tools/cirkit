from typing import cast

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import EmbeddingLayer, HadamardLayer, KroneckerLayer, SumLayer
from cirkit.symbolic.parameters import ConstantParameter, Parameter
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from cirkit.utils.scope import Scope


def cp(
    shape: tuple[int, ...],
    rank: int,
    *,
    factor_param: Parameterization | None = None,
    weight_param: Parameterization | None = None,
) -> Circuit:
    r"""Constructs a circuit encoding a CP factorization of an $n$-dimensional tensor.

    Formally, given the shape of a tensor $\mathcal{T}\in\mathbb{R}^{I_1\times \cdots\times I_n}$,
    this method returns a circuit $c$ over $n$ discrete random variables $\{X_j\}_{j=1}^n$,
    each taking value between $0$ and $I_j$ for $1\leq j\leq n$,
    and $c$ computes a rank-$R$ CP factorization, i.e.,

    $$
    c(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{i=1}^R a^{(1)}_{X_1 i} \cdots a^{(n)}_{X_n i},
    $$

    where for $1\leq j\leq n$ we have that $\mathbf{A}^{(j)}\in\mathbb{R}^{I_j\times R}$ is the $j$-th factor.

    Furthermore, this method allows you to return a circuit encoding a CP decomposition
    with additional weights, i.e., a CP factorization of the form

    $$
    c(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{i=1}^R w_i \: a^{(1)}_{X_1 i} \ldots a^{(n)}_{X_n i},
    $$

    where $\mathbf{w}\in\mathbb{R}^R$ are additional weights.

    This method allows you to specify different types of parameterizations for the factors and
    possibly the additional weights. For example, if the arguments ```factor_param``` and
    ```weight_param``` are both equal to a
    [parameterization][cirkit.templates.utils.Parameterization]
    ```Parameterization(activation="softmax", initialization="normal")```,
    then the returned circuit encodes a probabilistic model that is a mixture of fully-factorized
    models. That is, the returned circuit $c$ encodes the factorization of a non-negative tensor
    $\mathcal{T}\in\mathbb{R}_+^{I_1\times \ldots\times I_n}$ as the distribution

    $$
    p(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{i=1}^R p(Z=i) \: p(X_1\mid Z=i) \cdots p(X_n\mid Z=i),
    $$

    where $Z$ is a discrete latent variable modelled by $p(Z)$.

    Args:
        shape: The shape of the tensor to encode the CP factorization of.
        rank: The rank of the CP factorization. Defaults to 1.
        factor_param: The parameterization to use for the factor matrices.
            If None, then it defaults to no activation and uses an initialization based on
            independently sampling from a standard Gaussian distribution.
        weight_param: The parameterization to use for the weight coefficients.
            If None, then it defaults to fixed weights set all to one.

    Returns:
        Circuit: A circuit encoding a (possibly weighted) CP factorization.

    Raises:
        ValueError: If the given tensor shape is not valid.
        ValueError: If the rank is not a positive number.
    """
    if len(shape) < 1 or any(dim < 1 for dim in shape):
        raise ValueError("The tensor shape is not valid")
    if rank < 1:
        raise ValueError("The factorization rank should be a positive number")

    # Retrieve the factory to parameterize the embeddings
    if factor_param is None:
        factor_param = Parameterization(activation="none", initialization="normal")
    embedding_factory = parameterization_to_factory(factor_param)

    # Retrieve the sum layer weight, depending on whether we the CP factorization is weighted
    if weight_param is None:
        weight = Parameter.from_input(ConstantParameter(1, rank, value=1.0))
        weight_factory = None
    else:
        weight_factory = parameterization_to_factory(weight_param)
        weight = None

    # Construct the embedding, hadamard and sum layers
    embedding_layers = [
        EmbeddingLayer(Scope([i]), rank, 1, num_states=dim, weight_factory=embedding_factory)
        for i, dim in enumerate(shape)
    ]
    hadamard_layer = HadamardLayer(rank, arity=len(shape))
    sum_layer = SumLayer(rank, 1, arity=1, weight=weight, weight_factory=weight_factory)

    return Circuit(
        1,
        layers=embedding_layers + [hadamard_layer, sum_layer],
        in_layers={sum_layer: [hadamard_layer], hadamard_layer: embedding_layers},
        outputs=[sum_layer],
    )


def tucker(
    shape: tuple[int, ...],
    rank: int,
    *,
    factor_param: Parameterization | None = None,
    core_param: Parameterization | None = None,
) -> Circuit:
    r"""Constructs a circuit encoding a Tucker factorization of an $n$-dimensional tensor.

    Formally, given the shape of a tensor $\mathcal{T}\in\mathbb{R}^{I_1\times \cdots\times I_n}$,
    this method returns a circuit $c$ over $n$ discrete random variables $\{X_j\}_{j=1}^n$,
    each taking value between $0$ and $I_j$ for $1\leq j\leq n$,
    and $c$ computes a rank-$R$ Tucker factorization, i.e.,

    $$
    c(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{r_1=1}^R \cdots \sum_{r_n=1}^R w_{r_1\cdots r_n} a^{(1)}_{X_1 r_1} \cdots a^{(n)}_{X_n r_n},
    $$

    where for $1\leq j\leq n$ we have that $\mathbf{A}^{(j)}\in\mathbb{R}^{I_j\times R}$ is the $j$-th factor,
    and $\mathcal{W}\in\mathbb{R}^{R\times\cdots\times R}$ is an $n$-dimensional tensor, sometimes called
    the core tensor of the Tucker factorization.

    This method allows you to specify different types of parameterizations for the factors and
    possibly the additional weights. For example, if the arguments ```factor_param``` and
    ```weight_param``` are both equal to a
    [parameterization][cirkit.templates.utils.Parameterization]
    ```Parameterization(activation="softmax", initialization="normal")```,
    then the returned circuit encodes a probabilistic model that is a mixture of fully-factorized
    models. That is, the returned circuit $c$ encodes the factorization of a non-negative tensor
    $\mathcal{T}\in\mathbb{R}_+^{I_1\times \ldots\times I_n}$ as the distribution

    $$
    p(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{r_1=1}^R \cdots \sum_{r_n=1}^R p(Z=(r_1,\ldots,r_n)) \: p(X_1\mid Z=r_1) \cdots p(X_n\mid Z=r_n),
    $$

    where $Z$ is a discrete latent variable taking value in $\{1,\ldots,R\}^n$ and modelled by
    $p(Z)$.

    Args:
        shape: The shape of the tensor to encode the Tucker factorization of.
        rank: The rank of the Tucker factorization. Defaults to 1.
        factor_param: The parameterization to use for the factor matrices.
            If None, then it defaults to no activation and uses an initialization based on
            independently sampling from a standard Gaussian distribution.
        core_param: The parameterization to use for the core tensor.
            If None, then it defaults to no activation and uses an initialization based on
            independently sampling from a standard Gaussian distribution.

    Returns:
        Circuit: A circuit encoding a Tucker factorization.

    Raises:
        ValueError: If the given tensor shape is not valid.
        ValueError: If the rank is not a positive number.
    """
    if len(shape) < 1 or any(dim < 1 for dim in shape):
        raise ValueError("The tensor shape is not valid")
    if rank < 1:
        raise ValueError("The factorization rank should be a positive number")

    # Retrieve the factory to parameterize the embeddings and the core tensor
    if factor_param is None:
        factor_param = Parameterization(activation="none", initialization="normal")
    if core_param is None:
        core_param = Parameterization(activation="none", initialization="normal")
    embedding_factory = parameterization_to_factory(factor_param)
    weight_factory = parameterization_to_factory(core_param)

    # Construct the embedding, kronecker and sum layers
    embedding_layers = [
        EmbeddingLayer(Scope([i]), rank, 1, num_states=dim, weight_factory=embedding_factory)
        for i, dim in enumerate(shape)
    ]
    kronecker_layer = KroneckerLayer(rank, arity=len(shape))
    sum_layer = SumLayer(cast(int, rank ** len(shape)), 1, arity=1, weight_factory=weight_factory)

    return Circuit(
        1,
        layers=embedding_layers + [kronecker_layer, sum_layer],
        in_layers={sum_layer: [kronecker_layer], kronecker_layer: embedding_layers},
        outputs=[sum_layer],
    )
