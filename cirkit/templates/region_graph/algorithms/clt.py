from typing import Optional, List
from scipy import sparse as sp
import networkx as nx
import numpy as np
import torch


from cirkit.templates.region_graph import RegionGraph, RegionNode, PartitionNode


def tree2rg(tree: np.ndarray) -> RegionGraph:

    num_variables = len(tree)
    region_graph = RegionGraph()
    partitions: List[Optional[PartitionNode]] = [None] * num_variables
    for v in range(num_variables):
        cur_v, prev_v = v, tree[v]
        while prev_v != -1:
            if partitions[prev_v] is None:
                p_scope = {v, prev_v}
                partition_node = PartitionNode(p_scope)
                partitions[prev_v] = partition_node
            else:
                p_scope = set(partitions[prev_v].scope)
                p_scope = {v} | p_scope
                partition_node = PartitionNode(p_scope)
                partitions[prev_v] = partition_node
            cur_v, prev_v = prev_v, tree[cur_v]

    regions: List[Optional[RegionNode]] = [None] * num_variables
    for cur_v in range(num_variables):
        prev_v = tree[cur_v]
        leaf_region = RegionNode({cur_v})
        if partitions[cur_v] is None:
            if prev_v != -1:
                region_graph.add_edge(leaf_region, partitions[prev_v])
            regions[cur_v] = leaf_region
        else:
            region_graph.add_edge(leaf_region, partitions[cur_v])
            p_scope = partitions[cur_v].scope
            if regions[cur_v] is None:
                regions[cur_v] = RegionNode(set(p_scope))
            region_graph.add_edge(partitions[cur_v], regions[cur_v])
            if prev_v != -1:
                region_graph.add_edge(regions[cur_v], partitions[prev_v])

    region_graph.freeze()
    return region_graph


def maximum_spanning_tree(root: int, adj_matrix: torch.Tensor):
    mst = sp.csgraph.minimum_spanning_tree(-(adj_matrix.cpu().numpy() + 1.0), overwrite=True)
    bfs, tree = sp.csgraph.breadth_first_order(mst, directed=False, i_start=root, return_predecessors=True)
    tree[root] = -1
    return bfs, tree


def categorical_mutual_info(
    data: torch.LongTensor,
    alpha: float = 0.01,  # Laplace smoothing factor
    n_categories: Optional[int] = None,
    chunk_size: Optional[int] = None
):
    assert data.dtype == torch.long and data.ndim == 2
    n_samples, n_features = data.size()
    if n_categories is None:
        n_categories = int(data.max().item() + 1)
    if chunk_size is None:
        chunk_size = n_samples

    idx_features = torch.arange(0, n_features)
    idx_categories = torch.arange(0, n_categories)

    joint_counts = torch.zeros(n_features, n_features, n_categories ** 2, dtype=torch.long, device=data.device)
    for chunk in data.split(chunk_size):
        joint_values = chunk.t().unsqueeze(1) * n_categories + chunk.t().unsqueeze(0)
        joint_counts.scatter_add_(-1, joint_values.long(), torch.ones_like(joint_values))
    joint_counts = joint_counts.view(n_features, n_features, n_categories, n_categories)
    marginal_counts = joint_counts[idx_features, idx_features][:, idx_categories, idx_categories]

    marginals = (marginal_counts + n_categories * alpha) / (n_samples + n_categories ** 2 * alpha)
    joints = (joint_counts + alpha) / (n_samples + n_categories ** 2 * alpha)
    joints[idx_features, idx_features] = torch.diag_embed(marginals)  # Correct Laplace's smoothing for the marginals
    outers = torch.einsum('ik,jl->ijkl', marginals, marginals)

    return (joints * (joints.log() - outers.log())).sum(dim=(2, 3)).fill_diagonal_(0)


def learn_clt(
    data: torch.Tensor,
    leaf_type: str,
    chunk_size: Optional[int] = None,
    n_bins: Optional[int] = None,  # rescale categories in bins for ordinal features
    n_categories: Optional[int] = None
):
    if leaf_type in ['bernoulli', 'categorical']:
        if n_bins is not None:
            assert n_categories is not None, 'Number of categories must be known if rescaling in bins'
            data = torch.div(data, n_categories // n_bins, rounding_mode='floor')
        mutual_info = categorical_mutual_info(data.long(), n_categories=n_categories, chunk_size=chunk_size)
    elif leaf_type == 'gaussian':
        # todo: implement chunked computation
        mutual_info = (- 0.5 * torch.log(1 - torch.corrcoef(data.t()) ** 2))
    else:
        raise NotImplementedError('MI computation not implemented for %s leaves.' % leaf_type)

    bfs, tree = maximum_spanning_tree(root=0, adj_matrix=mutual_info)
    # use tree center too minimize tree depth
    nx_tree = nx.Graph([(node, parent) for node, parent in enumerate(tree) if parent != -1])
    bfs, tree = maximum_spanning_tree(root=nx.center(nx_tree)[0], adj_matrix=mutual_info)

    return tree
