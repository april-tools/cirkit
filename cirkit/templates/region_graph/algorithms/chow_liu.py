import numpy as np
import torch
from scipy import sparse as sp
from torch import Tensor

from cirkit.templates.region_graph.algorithms.utils import tree2rg
from cirkit.templates.region_graph.graph import RegionGraph


# pylint: disable-next=invalid-name
def ChowLiuTree(
    data: Tensor,
    input_type: str,
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
        input_type (str): The type of the input data, e.g. 'categorical', 'gaussian'.
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
    if input_type == "categorical":
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
    for chunk in data.split(chunk_size):  # type: ignore[no-untyped-call]
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
