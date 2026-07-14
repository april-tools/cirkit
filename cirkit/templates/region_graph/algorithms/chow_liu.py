from warnings import warn

import numpy as np
import torch
from scipy import sparse as sp
from torch import Tensor

from cirkit.templates.region_graph.algorithms.utils import tree2rg
from cirkit.templates.region_graph.graph import RegionGraph


# pylint: disable-next=invalid-name
def ChowLiuTree(
    data: Tensor,
    input_type: str | list[str],
    root: int | None = None,
    chunk_size: int | None = None,
    num_categories: int | None = None,
    cat_bins: int | None = None,
    as_region_graph: bool = True,
    heter_cont_bins: int | None = None,
    num_bins: int | None = None,
) -> np.ndarray | RegionGraph:
    """Learns a Chow-Liu Tree and returns it either as a
    list of predecessors (Bayesian net) or as region graph (HCLT).

    See:
        - *Approximating discrete probability distributions with dependence trees*
          [🔗](https://ieeexplore.ieee.org/abstract/document/1054142)
          CKCN Chow and Cong Liu.
          In IEEE transactions on Information Theory, 14(3):462–467, 1968b.

        - *What is the Relationship between Tensor Factorizations and Circuits (and How Can We
          Exploit it)?* [🔗](https://openreview.net/forum?id=Y7dRmpGiHj)
          Lorenzo Loconte and Antonio Mari and Gennaro Gala and Robert Peharz and Cassio de
          Campos and Erik Quaeghebeur and Gennaro Vessio and Antonio Vergari
          In Transactions on Machine Learning Research, 2025.

    Args:
        data (Tensor): The input data over which running the CLT algorithm,
            it must be in tabular form (i.e. a matrix).
        input_type (str | list): The type of the input data, e.g. 'categorical', 'gaussian'.
            If a list is provided, then each feature is treated differently according to its type.
        root (int | None): The index of the variable desired as root.
        chunk_size (int | None): Chunked computation, useful in case of large input data.
        num_categories (int | None): Specifies the number of categories in case of
            categorical data.
        cat_bins (int | None): In case of categorical input, it is used to rescale
            categories in bins for ordinal features, e.g. [0, 255] -> [0, 7],
            which is useful for images.
        as_region_graph (Optional[bool]): True to returns a region graph,
            False to return a list of predecessors. Defaults to True.
        heter_cont_bins (int | None): For heterogeneous (list) input, the number of bins used to
            discretize the continuous features so the whole MI matrix is estimated with the
            categorical estimator; if None, the mixed Gaussian/categorical estimator is used.
        num_bins: Deprecated parameters, equivalent to cat_bins

    Returns:
        A Chow-Liu Tree, either a list of predecessors or as a region graph.

    Raises:
        ValueError: If the number of categories has not been specified but the number of bins has.
        NotImplementedError: If the input type is neither 'categorical' nor 'gaussian'.
    """
    assert data.ndim == 2
    assert root is None or -1 < root < data.size(-1)

    if num_bins is not None:
        warn("Argument `num_bins` will soon be removed, you should use `cat_bins` instead.")
        assert cat_bins is None, "Cannot set both `num_bins` and `cat_bins`"
        cat_bins = num_bins

    if isinstance(input_type, list):
        is_categorical_mask = [name == "categorical" for name in input_type]
        mutual_info = (
            _heterogeneous_mutual_info(data, is_categorical_mask=is_categorical_mask)
            if heter_cont_bins is None
            else _heterogeneous_mutual_info_bin(
                data, is_categorical_mask=is_categorical_mask, bins=heter_cont_bins
            )
        )
    elif input_type == "categorical":
        if cat_bins is not None:
            if num_categories is None:
                raise ValueError("Number of categories must be known if rescaling in bins")
            data = torch.div(data, num_categories // cat_bins, rounding_mode="floor")
        mutual_info = _categorical_mutual_info(
            data.long(), num_categories=num_categories, chunk_size=chunk_size
        )
    elif input_type == "gaussian":
        mutual_info = _gaussian_mutual_info(data, chunk_size=chunk_size)
    else:
        raise NotImplementedError(f"MI computation not implemented for {input_type} input units")

    _, tree = _maximum_spanning_tree(adj_matrix=mutual_info, root=root)
    if as_region_graph:
        return tree2rg(tree)
    return tree


def _maximum_spanning_tree(
    adj_matrix: Tensor, root: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the maximum spanning tree of a given adjacency matrix rooted at a given variable.

    Args:
        adj_matrix (Tensor): The adjacency matrix.
        root (int | None): The index of the variable desired as root.
            If None, picks the one that minimizes depth.

    Returns:
        bfs: The BFS order of the spanning tree.
        tree: The spanning tree in form of list of predecessors.
    """
    mst = sp.csgraph.minimum_spanning_tree(-(adj_matrix.cpu().numpy() + 1.0), overwrite=True)
    if root is None:
        dist_from_all_nodes: np.ndarray = sp.csgraph.dijkstra(
            abs(mst).todense(), directed=False, return_predecessors=False
        )
        root = np.argmin(np.max(dist_from_all_nodes, axis=1)).item()
    bfs, tree = sp.csgraph.breadth_first_order(
        mst, directed=False, i_start=root, return_predecessors=True
    )
    tree[root] = -1
    return bfs, tree


def _categorical_mutual_info(
    data: Tensor,
    alpha: float = 0.01,
    num_categories: int | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Computes the mutual information (MI) matrix of a matrix of integers.

    Args:
        data (Tensor): The input data over which computing the MI matrix,
            it must be in tabular form (i.e. a matrix).
        alpha (Tensor): Laplace smoothing factor.
        num_categories (int | None): Specifies the number of categories.
        chunk_size (int | None): Chunked computation, useful in case of large input data.

    Returns:
        The mutual information matrix (main diagonal is 0).
    """
    assert data.dtype == torch.long and data.ndim == 2
    n_samples, n_features = data.size()
    if num_categories is None:
        num_categories = int(data.max().item() + 1)
    if chunk_size is None:
        chunk_size = n_samples

    idx_features = torch.arange(0, n_features)
    idx_categories = torch.arange(0, num_categories)

    joint_counts = torch.zeros(
        n_features, n_features, num_categories**2, dtype=torch.long, device=data.device
    )
    for chunk in data.split(chunk_size):
        joint_values = chunk.t().unsqueeze(1) * num_categories + chunk.t().unsqueeze(0)
        joint_counts.scatter_add_(-1, joint_values.long(), torch.ones_like(joint_values))
    joint_counts = joint_counts.view(n_features, n_features, num_categories, num_categories)
    marginal_counts = joint_counts[idx_features, idx_features][:, idx_categories, idx_categories]

    marginals = (marginal_counts + num_categories * alpha) / (n_samples + num_categories**2 * alpha)
    joints = (joint_counts + alpha) / (n_samples + num_categories**2 * alpha)
    joints[idx_features, idx_features] = torch.diag_embed(
        marginals
    )  # Correct Laplace's smoothing for the marginals
    outers = torch.einsum("ik,jl->ijkl", marginals, marginals)

    return (joints * (joints.log() - outers.log())).sum(dim=(2, 3)).fill_diagonal_(0)


def _gaussian_mutual_info(
    data: Tensor,
    chunk_size: int | None = None,
) -> Tensor:
    """Computes the mutual information (MI) matrix assuming jointly Gaussian variables.

    For a pair of jointly Gaussian variables the MI has the closed form
    I(X_i, X_j) = -0.5 * log(1 - rho_ij ** 2), where rho_ij is their Pearson
    correlation coefficient. The unnormalized covariance is accumulated over chunks
    of samples, this reduces the centered-data scratch space from
    O(n_samples * n_features) to O(chunk_size * n_features).

    Args:
        data (Tensor): The input data over which computing the MI matrix, it must be in
            tabular form, i.e., a matrix of shape (num_samples, num_features).
        chunk_size (int | None): Number of samples per chunk. If None, all samples are
            processed at once.

    Returns:
        Tensor: The mutual information matrix, with main diagonal equal to 0.

    Raises:
        ValueError: If data is not a real floating-point matrix with at least two samples
            and one feature, or if chunk_size is not a positive integer.
    """
    n_samples, n_features = data.size()

    assert not torch.is_complex(data)
    assert torch.is_floating_point(data)
    assert n_samples > 1
    assert n_features > 0

    if chunk_size is None:
        chunk_size = n_samples
    elif not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"Expected chunk_size to be a positive integer, but found {chunk_size}")

    gaussian_correlation_epsilon = torch.finfo(data.dtype).eps
    mean = data.mean(dim=0)
    covariance = torch.zeros(n_features, n_features, dtype=data.dtype, device=data.device)

    # Accumulate S = sum_n (x_n - mean)(x_n - mean)^T over sample chunks
    for chunk in data.split(chunk_size):
        centered = chunk - mean
        covariance = covariance + centered.t() @ centered

    variance = covariance.diagonal()
    inv_std = torch.zeros_like(variance)
    non_constant = variance > 0
    inv_std[non_constant] = variance[non_constant].rsqrt()

    # rho_ij = S_ij / sqrt(S_ii S_jj)
    correlation = covariance.mul(inv_std.unsqueeze(1)).mul(inv_std.unsqueeze(0))
    # I(X_i, X_j) = -0.5 * log(1 - rho_ij^2)
    squared_correlation = correlation.square().clamp_max(1.0 - gaussian_correlation_epsilon)
    mutual_info = -0.5 * torch.log1p(-squared_correlation)

    return mutual_info.fill_diagonal_(0)


def _heterogeneous_mutual_info(
    data: Tensor, is_categorical_mask: list[bool], normalize: bool = True
) -> Tensor:
    """Computes the mutual information (MI) matrix for heterogeneous data
    (both discrete/categorical data and continuous).
    The MI among continuous variables is computed as if they were a Multivariate Gaussian.
    The MI among discrete variables is computed using the categorical mutual information
    defined above.
    The MI between a continuous variable C and discrete variable D is computed using the formula:
        I(C, D) = H(C) - H(C | D)
    assuming gaussian distributions p(C|D) for continuous variables when conditioned on discrete
    variables and gaussian marginals p(c).

    Args:
        data: The input data over which computing the MI matrix,
            it must be in tabular form (i.e. a matrix).
        is_categorical_mask: A boolean mask of the same length as the number
            of columns in `data`, indicating if the column has to be considered categorical.
        A list of strings indicating the type of each variable whether each column in the
            data is categorical (True) or continuous (False).
        normalize: If True, normalizes the mutual information matrix by the entropy
            of each variable. NMI(X,Y) = 2 * I(X,Y) / (H(X) + H(Y)).

    Returns:
        The mutual information matrix (main diagonal is 0).
    """

    gaussian_entropy_epsilon = 1e-4

    is_categorical = torch.tensor(is_categorical_mask, dtype=torch.bool, device=data.device)
    continuous_subset = torch.where(~is_categorical)[0]
    discrete_subset = torch.where(is_categorical)[0]

    mi_matrix = torch.zeros((data.shape[1], data.shape[1]), dtype=torch.float32, device=data.device)

    # Compute mutual information for continuous variables as they were a Multivariate Gaussian
    if len(continuous_subset) > 1:
        mi_matrix[continuous_subset.unsqueeze(1), continuous_subset] = _gaussian_mutual_info(
            data[:, continuous_subset]
        ).float()

    # Compute mutual information for discrete variables
    if len(discrete_subset) > 1:
        mi_matrix[discrete_subset.unsqueeze(1), discrete_subset] = _categorical_mutual_info(
            data=data[:, discrete_subset].long(),
            num_categories=None,
            chunk_size=None,
        ).float()

    def gaussian_entropy(x: Tensor) -> Tensor:
        return 0.5 * (
            torch.log(2 * torch.pi * torch.var(x, unbiased=False) + gaussian_entropy_epsilon) + 1
        )

    # Precomputing number of categories for discrete variables
    num_categories = {
        d_index: int(data[:, d_index].max() + 1) for d_index in discrete_subset.tolist()
    }

    # Precomputing marginals p(D) for every discrete variable
    p_d = {
        d_index: data[:, d_index].long().bincount(minlength=num_categories[d_index]).float()
        / data.shape[0]
        for d_index in discrete_subset.tolist()
    }

    # precomputing gaussian entropy H(C) for each continuous variable
    h_c = {c_index: gaussian_entropy(data[:, c_index]) for c_index in continuous_subset.tolist()}

    # I(C, D) = H(C) - H(C | D)
    for c_index in continuous_subset.tolist():
        for d_index in discrete_subset.tolist():
            # H(C | D) = sum_D{ integral_C{ p(C|D)p(D) log_p(C|D) } } = sum_D{ -H[p(C|D)]p(D) }

            # Computing H[p(C|D)] for each category of D
            h_c_given_d = torch.stack(
                [
                    gaussian_entropy(data[:, c_index][data[:, d_index] == i])
                    for i in range(num_categories[d_index])
                ],
                dim=0,
            )

            # I(C, D) = H(C) - H(C | D) = H(C) - sum_D{ H[p(C|D)]p(D) }
            mi_matrix[c_index, d_index] = h_c[c_index] - torch.sum(h_c_given_d * p_d[d_index])
            mi_matrix[d_index, c_index] = mi_matrix[
                c_index, d_index
            ]  # mutual information is symmetric

    if normalize:
        # NMI(X, Y) = 2 * I(X, Y) / (H(X) + H(Y))
        entropy = torch.zeros(data.shape[1], dtype=torch.float32, device=data.device)
        entropy[continuous_subset] = torch.tensor(
            list(h_c.values()), dtype=torch.float32, device=data.device
        )
        entropy[discrete_subset] = torch.tensor(
            [-(p.log() * p).sum() for p in p_d.values()],
            dtype=torch.float32,
            device=data.device,
        )
        mi_matrix = 2 * mi_matrix / (entropy.unsqueeze(0) + entropy.unsqueeze(1))

    return mi_matrix.fill_diagonal_(0)


def _heterogeneous_mutual_info_bin(
    data: Tensor, is_categorical_mask: list[bool], bins: int = 10
) -> Tensor:
    """Computes the mutual information (MI) matrix for heterogeneous data by binning.

    Differently from `_heterogeneous_mutual_info`, which treats the continuous variables as a
    Multivariate Gaussian, this estimator discretizes each continuous variable into integer bins
    (by rescaling and rounding) and then computes the whole MI matrix with the categorical mutual
    information estimator. This avoids any Gaussian assumption on the continuous variables, at the
    cost of a binning approximation.

    Args:
        data: The input data over which computing the MI matrix,
            it must be in tabular form (i.e. a matrix).
        is_categorical_mask: A boolean mask of the same length as the number of columns in `data`,
            indicating if the column has to be considered categorical. The columns that are not
            categorical (i.e. continuous) are the ones being discretized into bins.
        bins: The number of bins into which each continuous variable is discretized.

    Returns:
        The mutual information matrix (main diagonal is 0).
    """

    is_categorical = torch.tensor(is_categorical_mask, dtype=torch.bool, device=data.device)
    continuous_subset = torch.where(~is_categorical)[0]

    x = data[:, continuous_subset].clone()
    x = (x - x.min(dim=0, keepdim=True).values) * bins / x.max(dim=0, keepdim=True).values
    x = x.round()

    discretized_data = data.clone()
    discretized_data[:, continuous_subset] = x

    return _categorical_mutual_info(discretized_data.long()).float()
