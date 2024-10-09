from collections import defaultdict
from typing import Optional, List
from scipy import sparse as sp
import networkx as nx
import numpy as np
import torch


from cirkit.templates.region_graph import RegionGraph, RegionNode, PartitionNode, RegionGraphNode


def tree2rg(tree: np.ndarray) -> RegionGraph:
    """Convert a tree structure into a region graph. Useful to convert CLTs into HCLT region graphs.
     More details in https://arxiv.org/abs/2409.07953.

    Args:
        tree (np.ndarray): A tree in form of list of predecessors, i.e. tree[i] is the parent of i,
        and is equal to -1 when i is root.

    Returns:
        A region graph.
    """
    num_variables = len(tree)
    root_region = RegionNode(range(num_variables))
    nodes: list[RegionGraphNode] = []
    in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = defaultdict(list)
    partitions: List[Optional[PartitionNode]] = [None] * num_variables

    for v in range(num_variables):
        cur_v, prev_v = v, tree[v]
        while prev_v != -1:
            if partitions[prev_v] is None:
                p_scope = {v, prev_v}
                partitions[prev_v] = PartitionNode(p_scope)
            else:
                p_scope = set(partitions[prev_v].scope)
                p_scope = {v} | p_scope
                partitions[prev_v] = PartitionNode(p_scope)
            cur_v, prev_v = prev_v, tree[cur_v]

    for part_node in partitions:
        if part_node is not None:
            nodes.append(part_node)

    regions: List[Optional[RegionNode]] = [None] * num_variables
    for cur_v in range(num_variables):
        prev_v = tree[cur_v]
        leaf_region = RegionNode({cur_v})
        nodes.append(leaf_region)
        if partitions[cur_v] is None:
            if prev_v != -1:
                in_nodes[partitions[prev_v]].append(leaf_region)
            regions[cur_v] = leaf_region
        else:
            in_nodes[partitions[cur_v]].append(leaf_region)
            p_scope = partitions[cur_v].scope
            if regions[cur_v] is None:
                regions[cur_v] = RegionNode(set(p_scope))
                nodes.append(regions[cur_v])
            in_nodes[regions[cur_v]].append(partitions[cur_v])
            if prev_v != -1:
                in_nodes[partitions[prev_v]].append(regions[cur_v])

    return RegionGraph(nodes, in_nodes, [root_region])


def maximum_spanning_tree(adj_matrix: torch.Tensor, root: int):
    """Runs the maximum spanning tree of an given adjacency matrix rooted at a given variable.

    Args:
        adj_matrix (Tensor): The adjacency matrix.
        root (int): The index of the variable desired as root.

    Returns:
        bfs: The BFS order of the spanning tree.
        tree: The spanning tree in form of list of predecessors.
    """
    mst = sp.csgraph.minimum_spanning_tree(-(adj_matrix.cpu().numpy() + 1.0), overwrite=True)
    bfs, tree = sp.csgraph.breadth_first_order(mst, directed=False, i_start=root, return_predecessors=True)
    tree[root] = -1
    return bfs, tree


def categorical_mutual_info(
    data: torch.LongTensor,
    alpha: float = 0.01,
    num_categories: Optional[int] = None,
    chunk_size: Optional[int] = None
):
    """Computes the mutual information matrix of a matrix of integers.

    Args:
        data (Tensor): The input data over which computing the MI matrix, it must be in tabular form (i.e. a matrix).
        alpha (Tensor): Laplace smoothing factor.
        num_categories (Optional[int]): Specifies the number of categories.
        chunk_size (Optional[int]): Chunked computation, useful in case of large input data.

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

    joint_counts = torch.zeros(n_features, n_features, num_categories ** 2, dtype=torch.long, device=data.device)
    for chunk in data.split(chunk_size):
        joint_values = chunk.t().unsqueeze(1) * num_categories + chunk.t().unsqueeze(0)
        joint_counts.scatter_add_(-1, joint_values.long(), torch.ones_like(joint_values))
    joint_counts = joint_counts.view(n_features, n_features, num_categories, num_categories)
    marginal_counts = joint_counts[idx_features, idx_features][:, idx_categories, idx_categories]

    marginals = (marginal_counts + num_categories * alpha) / (n_samples + num_categories ** 2 * alpha)
    joints = (joint_counts + alpha) / (n_samples + num_categories ** 2 * alpha)
    joints[idx_features, idx_features] = torch.diag_embed(marginals)  # Correct Laplace's smoothing for the marginals
    outers = torch.einsum('ik,jl->ijkl', marginals, marginals)

    return (joints * (joints.log() - outers.log())).sum(dim=(2, 3)).fill_diagonal_(0)


def learn_clt(
    data: torch.Tensor,
    input_type: str,
    root: Optional[int] = None,
    chunk_size: Optional[int] = None,
    num_categories: Optional[int] = None,
    num_bins: Optional[int] = None,
    as_region_graph: Optional[bool] = False
):
    """Learns a Chow-Liu Tree and returns it either as a list of predecessors (Bayesian net) or as region graph (HCLT).

    Details in https://arxiv.org/abs/2409.07953.

    Args:
        data (Tensor): The input data over which running the CLT algorithm, it must be in tabular form (i.e. a matrix).
        input_type (str): The type of the input data, e.g. 'categorical', 'gaussian'.
        root (Optional[int]): The index of the variable desired as root.
        chunk_size (Optional[int]): Chunked computation, useful in case of large input data.
        num_categories (Optional[int]): Specifies the number of categories in case of categorical data.
        num_bins (Optional[int]): In case of categorical input, it is used to rescale
         categories in bins for ordinal features, e.g. [0, 255] -> [0, 7], which is useful for images.
        as_region_graph (Optional[bool]): True to returns a region graph, False to return a list of predecessors.

    Returns:
        A CLT either a list of predecessors or as a region graph.
    """
    assert data.ndim == 2
    assert root is None or -1 < root < data.size(-1)
    if input_type == 'categorical':
        if num_bins is not None:
            assert num_categories is not None, 'Number of categories must be known if rescaling in bins'
            data = torch.div(data, num_categories // num_bins, rounding_mode='floor')
        mutual_info = categorical_mutual_info(data.long(), num_categories=num_categories, chunk_size=chunk_size)
    elif input_type == 'gaussian':
        # todo: implement chunked computation
        mutual_info = (- 0.5 * torch.log(1 - torch.corrcoef(data.t()) ** 2))
    else:
        raise NotImplementedError('MI computation not implemented for %s input units.' % input_type)

    if root is not None:
        bfs, tree = maximum_spanning_tree(adj_matrix=mutual_info, root=root)
    else:
        bfs, tree = maximum_spanning_tree(adj_matrix=mutual_info, root=0)
        # use tree center too minimize tree depth
        nx_tree = nx.Graph([(node, parent) for node, parent in enumerate(tree) if parent != -1])
        bfs, tree = maximum_spanning_tree(adj_matrix=mutual_info, root=nx.center(nx_tree)[0])

    if as_region_graph:
        return tree2rg(tree)
    else:
        return tree
