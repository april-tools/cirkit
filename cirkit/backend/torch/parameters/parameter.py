from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

import torch
from torch import Tensor

from cirkit.backend.torch.graph.folding import (
    AddressBook,
    AddressBookEntry,
    FoldIndexInfo,
    build_address_book_entry,
    build_address_book_stacked_entry,
    build_fold_index_info,
)
from cirkit.backend.torch.graph.modules import TorchDiAcyclicGraph
from cirkit.backend.torch.graph.nodes import AbstractTorchModule
from cirkit.backend.torch.graph.optimize import GraphOptEntry, GraphOptMatch, GraphOptPatternDefn


class TorchParameterNode(AbstractTorchModule, ABC):
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

    @torch.no_grad()
    def reset_parameters(self) -> None:
        ...


class TorchParameterLeaf(TorchParameterNode, ABC):
    @property
    def is_initialized(self) -> bool:
        return True

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

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *self.in_shapes, *self.config.items()

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


class TorchUnaryOpParameter(TorchParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...], *, num_folds: int = 1) -> None:
        super().__init__(in_shape, num_folds=num_folds)

    def __call__(self, x: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class TorchBinaryOpParameter(TorchParameterOp, ABC):
    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    def __call__(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x1, x2)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        ...


class ParameterAddressBook(AddressBook):
    def lookup(
        self, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Iterator[Tuple[Tensor, ...]]:
        # A useful function combining the modules outputs, and then possibly applying an index
        def select_index(mids: List[int], idx: Optional[Tensor]) -> Tensor:
            if len(mids) == 1:
                t = module_outputs[mids[0]]
            else:
                t = torch.cat([module_outputs[mid] for mid in mids], dim=0)
            return t if idx is None else t[idx]

        # Loop through the entries and yield inputs
        for entry in self._entries:
            in_module_ids = entry.in_module_ids

            # Catch the case there are some inputs coming from other modules
            if in_module_ids:
                x = tuple(
                    select_index(mids, in_idx)
                    for mids, in_idx in zip(in_module_ids, entry.in_fold_idx)
                )
                yield x
                continue

            # Catch the case there are no inputs coming from other modules
            yield ()

    @classmethod
    def from_index_info(
        cls, ordering: Iterable[TorchParameterNode], fold_idx_info: FoldIndexInfo
    ) -> "ParameterAddressBook":
        # The address book entries being built
        entries: List[AddressBookEntry] = []

        # A useful dictionary mapping module ids to their number of folds
        num_folds: Dict[int, int] = {}

        # Build the bookkeeping data structure by following the topological ordering
        for mid, m in enumerate(ordering):
            # Retrieve the index information of the input modules
            in_modules_fold_idx = fold_idx_info.in_fold_idx[mid]

            # Catch the case of a folded module having the input of the network as input
            if in_modules_fold_idx:
                entry = build_address_book_entry(in_modules_fold_idx, num_folds=num_folds)
            # Catch the case of a folded module without inputs
            else:
                entry = AddressBookEntry([], [])

            num_folds[mid] = m.num_folds
            entries.append(entry)

        # Append the last bookkeeping entry with the information to compute the output tensor
        entry = build_address_book_stacked_entry(
            [fold_idx_info.out_fold_idx], num_folds=num_folds, output=True
        )
        entries.append(entry)

        return ParameterAddressBook(entries)


class TorchParameter(TorchDiAcyclicGraph[TorchParameterNode]):
    @property
    def num_folds(self) -> int:
        return sum(n.num_folds for n in self.outputs)

    @property
    def shape(self) -> Tuple[int, ...]:
        return next(self.outputs).shape

    @classmethod
    def from_sequence(cls, p: "TorchParameter", *ns: TorchParameterNode) -> "TorchParameter":
        assert not p.has_address_book and not p._fold_idx_info
        nodes = p.nodes + list(ns)
        in_nodes = dict(p.nodes_inputs)
        out_nodes = dict(p.nodes_outputs)
        for i, n in enumerate(ns):
            in_nodes[n] = [ns[i - 1]] if i - 1 >= 0 else [p.output]
            out_nodes[n] = [ns[i + 1]] if i + 1 < len(ns) else []
        out_nodes[p.output] = [ns[0]]
        return TorchParameter(
            nodes, in_nodes, out_nodes, topologically_ordered=p.is_topologically_ordered
        )

    @classmethod
    def from_unary(cls, p: "TorchParameter", n: TorchUnaryOpParameter) -> "TorchParameter":
        return TorchParameter.from_sequence(p, n)

    @classmethod
    def from_binary(
        cls,
        p1: "TorchParameter",
        p2: "TorchParameter",
        n: TorchBinaryOpParameter,
    ) -> "TorchParameter":
        p_nodes = p1.nodes + p2.nodes + [n]
        in_nodes = {**p1.nodes_inputs, **p2.nodes_inputs}
        out_nodes = {**p1.nodes_outputs, **p2.nodes_outputs}
        in_nodes[n] = [p1.output, p2.output]
        out_nodes[p1.output] = [n]
        out_nodes[p2.output] = [n]
        return TorchParameter(
            p_nodes,
            in_nodes,
            out_nodes,
            topologically_ordered=p1.is_topologically_ordered and p2.is_topologically_ordered,
        )

    def _build_address_book(self) -> AddressBook:
        fold_idx_info = self._fold_idx_info
        if fold_idx_info is None:
            fold_idx_info = build_fold_index_info(
                self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
            )
        address_book = ParameterAddressBook.from_index_info(
            self.topological_ordering(), fold_idx_info
        )
        self._fold_idx_info = None
        return address_book

    def reset_parameters(self) -> None:
        """Reset the input parameters."""
        for p in self.nodes:
            p.reset_parameters()

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        return self._eval_forward()  # (F, d1, d2, ..., dk)


ParameterOptEntry = GraphOptEntry[TorchParameterNode]
ParameterOptPatternDefn = GraphOptPatternDefn[TorchParameterNode]
ParameterOptPattern = Type[ParameterOptPatternDefn]
ParameterOptMatch = GraphOptMatch[TorchParameterNode]
