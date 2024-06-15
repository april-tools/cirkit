from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.graph.modules import (
    AddressBook,
    FoldIndexInfo,
    TorchRootedDiAcyclicGraph,
    build_fold_index_info,
)
from cirkit.backend.torch.graph.nodes import TorchModule


class TorchParameterNode(TorchModule, ABC):
    """The abstract base class for all reparameterizations."""

    def __init__(self, *, num_folds: int = 1, **kwargs) -> None:
        """Init class."""
        super().__init__(num_folds=num_folds)

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


class ParameterAddressBook(AddressBook):
    def lookup_module_inputs(
        self, module_id: int, module_outputs: List[Tensor], *, in_network: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        # Retrieve the input tensor given by other modules
        entry = self._entries[module_id]
        in_module_ids = entry.in_module_ids

        # Catch the case there are no inputs coming from other modules
        if not in_module_ids:
            return ()

        # Catch the case there are some inputs coming from other modules
        in_tensors = tuple(
            torch.cat(tuple(module_outputs[mid] for mid in mids), dim=0)
            for mids in entry.in_module_ids
        )
        x = tuple(
            t[in_idx] if in_idx is not None else t
            for t, in_idx in zip(in_tensors, entry.in_fold_idx)
        )
        return x

    def lookup_output(self, module_outputs: List[Tensor]) -> Tensor:
        outputs = self.lookup_module_inputs(-1, module_outputs=module_outputs)
        output = torch.cat(outputs, dim=0)
        (in_fold_idx,) = self._entries[-1].in_fold_idx
        if in_fold_idx is not None:
            output = output[in_fold_idx]
        return output

    @classmethod
    def from_index_info(cls, fold_idx_info: FoldIndexInfo) -> "ParameterAddressBook":
        ...


class TorchParameter(TorchRootedDiAcyclicGraph[TorchParameterNode]):
    def __init__(
        self,
        nodes: List[TorchParameterNode],
        in_nodes: Dict[TorchParameterNode, List[TorchParameterNode]],
        out_nodes: Dict[TorchParameterNode, List[TorchParameterNode]],
        *,
        topologically_ordered: bool = False,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ) -> None:
        super().__init__(nodes, in_nodes, out_nodes, topologically_ordered=topologically_ordered)
        self._fold_idx_info = fold_idx_info

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

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        y = self._eval_forward()  # (F, d1, d2, ..., dk)
        return y

    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        return ParameterAddressBook.from_index_info(fold_idx_info)
