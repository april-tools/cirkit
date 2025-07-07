from collections import defaultdict
from typing import cast

import numpy as np
from scipy import linalg

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import EmbeddingLayer, HadamardLayer, KroneckerLayer, Layer, SumLayer
from cirkit.symbolic.parameters import ConstantParameter, Parameter, ParameterFactory
from cirkit.templates.utils import (
    InputLayerFactory,
    Parameterization,
    name_to_input_layer_factory,
    named_parameterizations_to_factories,
    parameterization_to_factory,
)
from cirkit.utils.scope import Scope


def _input_layer_factory_builder(
    input_layer: str, dim: int, factor_param_kwargs: dict[str, ParameterFactory]
) -> InputLayerFactory:
    match input_layer:
        case "categorical":
            factor_dim_kwargs = {"num_categories": dim}
        case "binomial":
            factor_dim_kwargs = {"total_count": dim}
        case "embedding":
            factor_dim_kwargs = {"num_states": dim}
        case _:
            assert False
    return name_to_input_layer_factory(input_layer, **factor_dim_kwargs, **factor_param_kwargs)


def cp(
    shape: tuple[int, ...],
    rank: int,
    *,
    input_layer: str = "embedding",
    input_params: dict[str, Parameterization] | None = None,
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

    where for $1\leq j\leq n$ we have that $\mathbf{A}^{(j)}\in\mathbb{R}^{I_j\times R}$ is the
    $j$-th factor.

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
        input_layer: The input layer to use for the factors. It can be 'embedding', 'categorical'
            or 'binomial'. Defaults to 'embedding'. If it is 'embedding' then it corresponds to the
            CP factorization described above where the factors are matrices.
        input_params: A dictionary mapping each name of a parameter of the input layer to
            its parameterization. If it is None and ```input_layer``` is 'embedding', then
            it defaults to no activation and uses an initialization based on
            independently sampling from a standard Gaussian distribution.
        weight_param: The parameterization to use for the weight coefficients.
            If None, then it defaults to fixed weights set all to one.

    Returns:
        Circuit: A circuit encoding a (possibly weighted) CP factorization.

    Raises:
        ValueError: If the given tensor shape is not valid.
        ValueError: If the rank is not a positive number.
        ValueError: If the input layer is not valid.
    """
    if len(shape) < 1 or any(dim < 1 for dim in shape):
        raise ValueError("The tensor shape is not valid")
    if rank < 1:
        raise ValueError("The factorization rank should be a positive number")
    if input_layer not in ["categorical", "binomial", "embedding"]:
        raise ValueError(f"The input layer {input_layer} is not valid for CP")

    # Retrieve the sum layer weight, depending on whether we the CP factorization is weighted
    if weight_param is None:
        weight = Parameter.from_input(ConstantParameter(1, rank, value=1.0))
        weight_factory = None
    else:
        weight_factory = parameterization_to_factory(weight_param)
        weight = None

    # Construct the factor, hadamard and sum layers
    if input_params is None:
        factor_param_kwargs = {}
    else:
        factor_param_kwargs = named_parameterizations_to_factories(input_params)
    embedding_layer_factories: list[InputLayerFactory] = [
        _input_layer_factory_builder(input_layer, dim, factor_param_kwargs) for dim in shape
    ]
    embedding_layers = [f(Scope([i]), rank) for i, f in enumerate(embedding_layer_factories)]
    hadamard_layer = HadamardLayer(rank, arity=len(shape))
    sum_layer = SumLayer(rank, 1, arity=1, weight=weight, weight_factory=weight_factory)

    return Circuit(
        layers=embedding_layers + [hadamard_layer, sum_layer],
        in_layers={sum_layer: [hadamard_layer], hadamard_layer: embedding_layers},
        outputs=[sum_layer],
    )


def tucker(
    shape: tuple[int, ...],
    rank: int,
    *,
    input_layer: str = "embedding",
    input_params: dict[str, Parameterization] | None = None,
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
    the core tensor. For example, if the arguments ```factor_param``` and
    ```core_param``` are both equal to a
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
        input_layer: The input layer to use for the factors. It can be 'embedding', 'categorical'
            or 'binomial'. Defaults to 'embedding'. If it is 'embedding' then it corresponds to the
            CP factorization described above where the factors are matrices.
        input_params: A dictionary mapping each name of a parameter of the input layer to
            its parameterization. If it is None and ```input_layer``` is 'embedding', then
            it defaults to no activation and uses an initialization based on
            independently sampling from a standard Gaussian distribution.
        core_param: The parameterization to use for the core tensor.
            If None, then it defaults to no activation and uses an initialization based on
            independently sampling from a standard Gaussian distribution.

    Returns:
        Circuit: A circuit encoding a Tucker factorization.

    Raises:
        ValueError: If the given tensor shape is not valid.
        ValueError: If the rank is not a positive number.
        ValueError: If the input layer is not valid.
    """
    if len(shape) < 1 or any(dim < 1 for dim in shape):
        raise ValueError("The tensor shape is not valid")
    if rank < 1:
        raise ValueError("The factorization rank should be a positive number")
    if input_layer not in ["categorical", "binomial", "embedding"]:
        raise ValueError(f"The input layer {input_layer} is not valid for Tucker")

    # Retrieve the factory to parameterize the core tensor
    if core_param is None:
        core_param = Parameterization(activation="none", initialization="normal")
    weight_factory = parameterization_to_factory(core_param)

    # Construct the embedding, kronecker and sum layers
    if input_params is None:
        factor_param_kwargs = {}
    else:
        factor_param_kwargs = named_parameterizations_to_factories(input_params)
    embedding_layer_factories: list[InputLayerFactory] = [
        _input_layer_factory_builder(input_layer, dim, factor_param_kwargs) for dim in shape
    ]
    embedding_layers = [f(Scope([i]), rank) for i, f in enumerate(embedding_layer_factories)]
    kronecker_layer = KroneckerLayer(rank, arity=len(shape))
    sum_layer = SumLayer(cast(int, rank ** len(shape)), 1, arity=1, weight_factory=weight_factory)

    return Circuit(
        layers=embedding_layers + [kronecker_layer, sum_layer],
        in_layers={sum_layer: [kronecker_layer], kronecker_layer: embedding_layers},
        outputs=[sum_layer],
    )


def tensor_train(
    shape: tuple[int, ...],
    rank: int,
    *,
    factor_param: Parameterization | None = None,
) -> Circuit:
    r"""Constructs a circuit encoding a Tensor-Train (TT) factorization of an $n$-dimensional
    tensor. This factorization is also called Matrix-Product State (MPS) in quantum physics.
    Note that the obtained circuit encodes the complete left-to-right contraction of the
    Note that the obtained circuit encodes the complete left-to-right contraction of the
    TT/MPS factorization, given an entry of the tensor being factorized.

    Formally, given the shape of a tensor $\mathcal{T}\in\mathbb{R}^{I_1\times \cdots\times I_n}$,
    this method returns a circuit $c$ over $n$ discrete random variables $\{X_j\}_{j=1}^n$,
    each taking value between $0$ and $I_j$ for $1\leq j\leq n$,
    and $c$ computes a rank-$R$ TT/MPS factorization, i.e.,

    $$
    c(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{r_1=1}^R \cdots \sum_{r_{n-1}=1}^R v^{(1)}_{X_1 r_1} v^{(2)}_{X_2 r_1 r_2} \cdots v^{(n-1)}_{X_{n-1} r_{n-2} r_{n-1}} v^{(n)}_{X_n r_{n-1}},  pylint: disable=line-too-long
    $$

    where $\mathbf{V}^{(1)}\in\mathbb{R}^{I_1\times R}$,
    $\mathbf{V}^{(n)}\in\mathbb{R}^{I_n\times R}$,
    and $\mathbf{V}^{(j)}\in\mathbb{R}^{I_j\times R\times R}$ for $1< j< n$
    are the factor tensors of the TT/MPS factorization.

    This method allows you to specify different types of parameterizations for the factor tensors.
    For instance, if the argument ```factor_param``` is equal to
    [parameterization][cirkit.templates.utils.Parameterization]
    ```Parameterization(dtype="complex")```
    then the returned circuit has complex parameters and therefore can be used
    to represent a many-body quantum system.

    Args:
        shape: The shape of the tensor to encode the TT/MPS factorization of.
        rank: The rank of the TT/MPS factorization. Defaults to 1.
        factor_param: The parameterization to use for the factor tensors.
            If None, then it defaults to no activation and uses an initialization based on
            independently sampling from a standard Gaussian distribution.

    Returns:
        Circuit: A circuit encoding a TT/MPS factorization.

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
    embedding_factory = parameterization_to_factory(factor_param)

    # Construct the first, last, and inner embedding layers
    first_embedding = EmbeddingLayer(
        Scope([0]), rank, num_states=shape[0], weight_factory=embedding_factory
    )
    last_embedding = EmbeddingLayer(
        Scope([len(shape) - 1]), rank, num_states=shape[-1], weight_factory=embedding_factory
    )
    inner_embeddings = [
        [
            EmbeddingLayer(Scope([i]), rank, num_states=dim, weight_factory=embedding_factory)
            for _ in range(rank)
        ]
        for i, dim in enumerate(shape[1:-1], start=1)
    ]

    # The inner sum layers will have a constant parameter matrix used to encode a matrix-vector
    # product, while the last sum layer will have a constant parameter matrix of ones used to
    # encode a vector dot product
    dot_ones = np.ones((1, rank))
    mav_ones = linalg.block_diag(*((dot_ones,) * rank))

    # Build the layers encoding the left-to-right contraction of the TT/MPS factorization
    layers: list[Layer] = [first_embedding, last_embedding] + [
        sl for sls in inner_embeddings for sl in sls
    ]
    in_layers: dict[Layer, list[Layer]] = defaultdict(list)
    cur_sl: Layer = first_embedding
    for i in range(len(shape) - 1):
        if i == len(shape) - 2:
            # i = n
            # Encode the vector dot product by stacking an hadamard layer and a sum layer
            prod_sl = HadamardLayer(rank, arity=2)
            sum_sl = SumLayer(
                rank,
                1,
                arity=1,
                weight=Parameter.from_input(ConstantParameter(1, rank, value=dot_ones)),
            )
            layers.append(prod_sl)
            layers.append(sum_sl)
            in_layers[sum_sl] = [prod_sl]
            in_layers[prod_sl] = [cur_sl, last_embedding]
            cur_sl = sum_sl
            continue
        # 0<= i< n
        # Encode the matrix-vector product by stacking hadamard layers and a sum layer
        prod_sls = [HadamardLayer(rank, arity=2) for _ in range(rank)]
        sum_sl = SumLayer(
            rank,
            rank,
            arity=rank,
            weight=Parameter.from_input(ConstantParameter(rank, rank * rank, value=mav_ones)),
        )
        layers.extend(prod_sls)
        layers.append(sum_sl)
        in_layers[sum_sl] = prod_sls
        for prod_sl, emb_sl in zip(prod_sls, inner_embeddings[i]):
            in_layers[prod_sl] = [cur_sl, emb_sl]
        cur_sl = sum_sl

    # Instantiate and return the circuit
    return Circuit(
        layers=layers,
        in_layers=in_layers,
        outputs=[cur_sl],
    )
