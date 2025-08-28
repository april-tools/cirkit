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
    num_bins: int | None = None,
    as_region_graph: bool = True,
) -> np.ndarray | RegionGraph:
    """Learns a Chow-Liu Tree and returns it either as a
    list of predecessors (Bayesian net) or as region graph (HCLT).

    Details in https://arxiv.org/abs/2409.07953.

    Args:
        data (Tensor): The input data over which running the CLT algorithm,
            it must be in tabular form (i.e. a matrix).
        input_type (str | list): The type of the input data, e.g. 'categorical', 'gaussian'.
            If a list is provided, then each feature is treated differently according to its type.
        root (int | None): The index of the variable desired as root.
        chunk_size (int | None): Chunked computation, useful in case of large input data.
        num_categories (int | None): Specifies the number of categories in case of
            categorical data.
        num_bins (int | None): In case of categorical input, it is used to rescale
            categories in bins for ordinal features, e.g. [0, 255] -> [0, 7],
            which is useful for images.
        as_region_graph (Optional[bool]): True to returns a region graph,
            False to return a list of predecessors. Defaults to True.

    Returns:
        A Chow-Liu Tree, either a list of predecessors or as a region graph.

    Raises:
        ValueError: If the number of categories has not been specified but the number of bins has.
        NotImplementedError: If the input type is neither 'categorical' nor 'gaussian'.
    """
    assert data.ndim == 2
    assert root is None or -1 < root < data.size(-1)
    if isinstance(input_type, list):
        mutual_info = _heterogeneous_mutual_info(
            data, is_categorical_mask=[name == "categorical" for name in input_type]
        )
    elif input_type == "categorical":
        if num_bins is not None:
            if num_categories is None:
                raise ValueError("Number of categories must be known if rescaling in bins")
            data = torch.div(data, num_categories // num_bins, rounding_mode="floor")
        mutual_info = _categorical_mutual_info(
            data.long(), num_categories=num_categories, chunk_size=chunk_size
        )
    elif input_type == "gaussian":
        # todo: implement chunked computation
        mutual_info = -0.5 * torch.log(1 - torch.corrcoef(data.t()) ** 2)
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


def _heterogeneous_mutual_info(
    data: Tensor, is_categorical_mask: list[bool], normalize: bool = True
) -> Tensor:
    """Computes the mutual information (MI) matrix for heterogeneous data
    (both discrete/categorical data and continuous).
    The MI among continuous variables is computed as if they were a Multivariate Gaussian.
    The MI among discrete variables is computed using the categorical mutual information defined above.
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
        mi_matrix[continuous_subset.unsqueeze(1), continuous_subset] = (
            -0.5
            * torch.log(
                1 - torch.corrcoef(data[:, continuous_subset].t()).fill_diagonal_(0) ** 2
            ).float()
        )

    # Compute mutual information for discrete variables
    if len(discrete_subset) > 1:
        mi_matrix[discrete_subset.unsqueeze(1), discrete_subset] = _categorical_mutual_info(
            data=data[:, discrete_subset].long(), num_categories=None, chunk_size=None
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
            [-(p.log() * p).sum() for p in p_d.values()], dtype=torch.float32, device=data.device
        )
        mi_matrix = 2 * mi_matrix / (entropy.unsqueeze(0) + entropy.unsqueeze(1))

    return mi_matrix.fill_diagonal_(0)
