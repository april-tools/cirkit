from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph import TorchRootedDiAcyclicGraph, build_unfolded_address_book, \
    build_folded_address_book, FoldAddressBook


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
        in_fold_idx: Optional[Dict[TorchParameterNode, List[List[Tuple[int, int]]]]] = None,
        out_fold_idx: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(nodes, in_nodes, out_nodes, topologically_ordered=topologically_ordered)

        # Build the bookkeeping data structure
        assert (in_fold_idx is None and out_fold_idx is None) or (
                in_fold_idx is not None and out_fold_idx is not None
        )
        if in_fold_idx is None:
            self._book = self._build_unfolded_bookkeeping()
        else:
            self._book = self._build_folded_bookkeeping(in_fold_idx, out_fold_idx)

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

    def _build_unfolded_bookkeeping(self) -> FoldAddressBook:
        return build_unfolded_address_book(
            self.topological_ordering(),
            incomings_fn=self.node_inputs,
            outputs=iter([self.output])
        )

    def _build_folded_bookkeeping(
        self,
        in_fold_idx: Dict[TorchParameterNode, List[List[Tuple[int, int]]]],
        out_fold_idx: List[Tuple[int, int]],
    ) -> FoldAddressBook:
        return build_folded_address_book(
            self.topological_ordering(),
            incomings_fn=self.node_inputs,
            num_folds_fn=lambda n: n.num_folds,
            in_fold_idx=in_fold_idx,
            out_fold_idx=out_fold_idx,
            in_stack=False
        )

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        outputs = []
        for entry, node in zip(self._book[:-1], self.nodes):
            in_node_ids, in_fold_idx = entry.in_module_ids, entry.in_fold_idx
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
        entry = self._book[-1]
        (out_node_ids,), (out_fold_idx,) = entry.in_module_ids, entry.in_fold_idx
        out_tensors = [outputs[i] for i in out_node_ids]
        if len(out_tensors) > 1:
            outputs = torch.cat(out_tensors, dim=0)
        else:
            (outputs,) = out_tensors
        if out_fold_idx is not None:
            outputs = outputs[out_fold_idx]
        return outputs
