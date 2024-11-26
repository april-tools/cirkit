from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import EmbeddingLayer, HadamardLayer, SumLayer
from cirkit.symbolic.parameters import ConstantParameter, Parameter
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from cirkit.utils.scope import Scope


def cp(
    shape: tuple[int, ...],
    rank: int,
    *,
    param: Parameterization | None = None,
    weighted: bool = False
) -> Circuit:
    """Constructs a circuit encoding a CP factorization of an n-dimensional tensor.

    Formally, given the shape of a tensor $\mathcal{T}\in\mathbb{R}^{I_1\times \ldots\times I_n}$,
    this method returns a circuit $c$ over $n$ discrete random variables $\{X_j\}_{j=1}^n$,
    each taking value between $0$ and $I_j$ for $1\leq j\leq n$,
    and $c$ computes a rank-$R$ CP factorization, i.e.,
    $$
    c(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{i=1}^R a^{(1)}_{X_1 i} \ldots a^{(n)}_{X_n i},
    $$
    where for $1\leq j\leq n$ we have that $\vA^{(j)}\in\bbR^{I_j\times R}$ is the $j$-th factor.

    Furthermore, this method allows you to return a circuit encoding a CP decomposition
    with additional weights, i.e., a CP factorization of the form
    $$
    c(X_1,\ldots,X_n) = t_{X_1\cdots X_n} = \sum_{i=1}^R w_i \: a^{(1)}_{X_1 i} \ldots a^{(n)}_{X_n i},
    $$
    where $\mathbf{w}\in\bbR^R$ are additional weights.

    This method allows you to specify different types of parameterizations for the factors and
    possibly the additional weights. For example, if the argument ```param``` is equal to a
    [parameterization][cirkit.templates.utils.Parameterization]
    ```Parameterization(activation="softmax", initialization="normal")```, and the argument
    ```weighted``` is set to True, then the returned circuit encodes a probabilistic model
    that is a mixture of fully-factorized models. That is, the returned circuit $c$ encodes
    the factorization of a non-negative tensor
    $\mathcal{T}\in\mathbb{R}_+^{I_1\times \ldots\times I_n}$ as the distribution
    $$
    p(X_1,\ldots,X_n) = t_{X_1\codts X_n} = \sum_{i=1}^R p(Z=i) \: p(X_1\mid Z=i) \cdots p(X_n\mid Z=i),
    $$
    where $Z$ is a discrete latent variable modelled by $p(Z)$.

    Args:
        shape: The shape of the tensor to encode the CP factorization of.
        rank: The rank of the CP factorization. Defaults to 1.
        param: The parameterization to use for the factor matrices, and for the
            additional weights of the CP factorization if ```weighted``` is True.
        weighted: Whether to construct a circuit encoding the weighted CP factorization.
            Defaults to False.

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

    # Retrieve the weight factory to parameterize the embeddings and
    # possibly the sum layer weight
    if param is None:
        param = Parameterization(activation="none", initialization="normal")
    weight_factory = parameterization_to_factory(param)

    # Retrieve the sum layer weight, depending on whether we want a weighted CP factorization
    sum_weight = weight_factory((1, rank)) if weighted else Parameter.from_input(
        ConstantParameter(1, rank, value=1.0)
    )

    # Construct the embedding, hadamard and sum layers
    embedding_layers = [
        EmbeddingLayer(Scope([i]), rank, 1, num_states=dim, weight_factory=weight_factory)
        for i, dim in enumerate(shape)
    ]
    hadamard_layer = HadamardLayer(rank, arity=len(shape))
    sum_layer = SumLayer(rank, 1, arity=1, weight=sum_weight)

    return Circuit(
        1, layers=embedding_layers + [hadamard_layer, sum_layer],
        in_layers={sum_layer: [hadamard_layer], hadamard_layer: embedding_layers},
        outputs=[sum_layer],
    )
