from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph import TorchRootedDiAcyclicGraph


class TorchParameterNode(nn.Module, ABC):
    """The abstract base class for all reparameterizations."""

    def __init__(self, *, num_folds: int = 1, **kwargs) -> None:
        """Init class."""
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return {}


class TorchParameterLeaf(TorchParameterNode, ABC):
    @property
    def is_initialized(self) -> bool:
        return True

    def initialize_(self) -> None:
        pass

    def __call__(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        ...


class TorchParameterOp(TorchParameterNode, ABC):
    def __init__(self, *in_shape: Tuple[int, ...], num_folds: int = 1):
        super().__init__(num_folds=num_folds)
        self.in_shapes = in_shape

    def __call__(self, *xs: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(*xs)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, *xs: Tensor) -> Tensor:
        ...


class TorchParameter(TorchRootedDiAcyclicGraph[TorchParameterNode]):
    def __init__(
        self,
        nodes: List[TorchParameterNode],
        in_nodes: Dict[TorchParameterNode, List[TorchParameterNode]],
        out_nodes: Dict[TorchParameterNode, List[TorchParameterNode]],
        *,
        topologically_ordered: bool = False,
        fold_in_nodes_idx: Optional[Dict[TorchParameterNode, List[List[Tuple[int, int]]]]] = None,
        fold_out_nodes_idx: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(nodes, in_nodes, out_nodes, topologically_ordered=topologically_ordered)

        # Build the bookkeeping data structure
        assert (fold_in_nodes_idx is None and fold_out_nodes_idx is None) or (
            fold_in_nodes_idx is not None and fold_out_nodes_idx is not None
        )
        if fold_in_nodes_idx is None:
            self._bookkeeping = self._build_unfolded_bookkeeping()
        else:
            self._bookkeeping = self._build_folded_bookkeeping(
                fold_in_nodes_idx, fold_out_nodes_idx
            )

    @property
    def num_folds(self) -> int:
        return self.output.num_folds

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.output.shape

    def initialize_(self) -> None:
        """Reset the input parameters."""
        # TODO: assuming parameter operators do not have any learnable parameters
        for p in self.inputs:
            p.initialize_()

    def _build_unfolded_bookkeeping(self) -> List[Tuple[List[List[int]], List[Optional[Tensor]]]]:
        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[List[int]], List[Optional[Tensor]]]] = []

        # A map from parameter nodes to their ids
        node_ids: Dict[TorchParameterNode, int] = {}

        # Build the bookkeeping data structure
        for p in self.topological_ordering():
            in_node_ids = [[node_ids[pi]] for pi in self.node_inputs(p)]
            bookkeeping_entry = (in_node_ids, [None] * len(in_node_ids))
            bookkeeping.append(bookkeeping_entry)
            node_ids[p] = len(node_ids)

        # Append a last bookkeeping entry with the info to extract the output
        out_nodes_ids = [node_ids[self.output]]
        bookkeeping_entry = ([out_nodes_ids], [None])
        bookkeeping.append(bookkeeping_entry)
        return bookkeeping

    def _build_folded_bookkeeping(
        self,
        fold_in_nodes_idx: Dict[TorchParameterNode, List[List[Tuple[int, int]]]],
        fold_out_nodes_idx: List[Tuple[int, int]],
    ) -> List[Tuple[List[List[int]], List[Optional[Tensor]]]]:
        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[List[int]], Optional[List[Tensor]]]] = []

        # A useful data dictonary mapping parameter node ids to their number of folds
        num_folds_map: Dict[int, int] = {}

        # Build the bookkeeping data structure
        # Note that the parameter nodes are already given in a topological ordering
        for p in self.topological_ordering():
            # Retrieve the index information from the folded parameter node
            in_nodes_idx = fold_in_nodes_idx[p]

            # Catch the case of the folded parameter node being a leaf one
            if in_nodes_idx is None:
                bookkeeping_entry = ([], None)
                num_folds_map[len(bookkeeping)] = p.num_folds
                bookkeeping.append(bookkeeping_entry)
                continue

            # Transpose the index information.
            # Since we cannot stack multiple inputs in a single tensor, as they might have different shapes
            # (e.g., think about a kronecker parameter node), we construct a bookkeeping entry for
            # each of the folded input to the folded parameter node.
            in_nodes_idx = list(map(list, zip(*in_nodes_idx)))

            # Retrieve the unique fold indices that reference the parameter node inputs
            in_nodes_ids = [sorted(list(set(si[0] for si in in_fi))) for in_fi in in_nodes_idx]

            # Compute the cumulative indices of the folded inputs
            cum_folded_node_ids_maps = [
                dict(zip(n_ids, np.cumsum([0] + [num_folds_map[pi] for pi in n_ids]).tolist()))
                for n_ids in in_nodes_ids
            ]

            # Build the bookkeeping entry
            in_fold_idx: List[Tensor] = []
            for i, in_fi in enumerate(in_nodes_idx):
                in_slice_idx: List[int] = []
                for si in in_fi:
                    in_slice_idx.append(cum_folded_node_ids_maps[i][si[0]] + si[1])
                in_slice_idx_t = torch.tensor(in_slice_idx)
                in_fold_idx.append(in_slice_idx_t)
            bookkeeping_entry = (in_nodes_ids, in_fold_idx)
            num_folds_map[len(bookkeeping)] = p.num_folds
            bookkeeping.append(bookkeeping_entry)

        # Append a last bookkeeping entry with the info to extract the output
        out_nodes_ids = sorted(list(set(si[0] for si in fold_out_nodes_idx)))
        cum_folded_nodes_ids_map = dict(
            zip(
                out_nodes_ids, np.cumsum([0] + [num_folds_map[li] for li in out_nodes_ids]).tolist()
            )
        )
        out_fold_idx: List[int] = [
            cum_folded_nodes_ids_map[si[0]] + si[1] for si in fold_out_nodes_idx
        ]
        if out_fold_idx == list(range(len(fold_out_nodes_idx))):
            out_fold_idx_t = None
        else:
            out_fold_idx_t = torch.tensor(out_fold_idx)
        bookkeeping_entry = ([out_nodes_ids], [out_fold_idx_t])
        bookkeeping.append(bookkeeping_entry)
        return bookkeeping

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        outputs = []
        for (in_node_ids, in_fold_idx), node in zip(self._bookkeeping[:-1], self.nodes):
            in_tensors = [[outputs[i] for i in input_idx] for input_idx in in_node_ids]
            if in_tensors:
                inputs = [
                    torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
                    for tensors in in_tensors
                ]
                inputs = tuple(
                    in_x if in_idx is None else in_x[in_idx]
                    for in_x, in_idx in zip(inputs, in_fold_idx)
                )
                pout = node(*inputs)
            else:
                pout = node()
            outputs.append(pout)

        # Retrieve the indices of the output tensors
        (out_nodes_ids,), (out_fold_idx,) = self._bookkeeping[-1]
        out_tensors = [outputs[i] for i in out_nodes_ids]
        if len(out_tensors) > 1:
            outputs = torch.cat(out_tensors, dim=0)
        else:
            (outputs,) = out_tensors
        if out_fold_idx is not None:
            outputs = outputs[out_fold_idx]
        return outputs
