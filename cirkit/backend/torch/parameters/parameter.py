import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy as shallowcopy
from functools import reduce
from itertools import chain
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from cirkit.backend.torch.graph.address_book import AddressBook, AddressBookEntry, FoldIndexInfo
from cirkit.backend.torch.graph.folding import (
    build_address_book_entry,
    build_address_book_stacked_entry,
    build_fold_index_info,
)
from cirkit.backend.torch.graph.modules import TorchDiAcyclicGraph
from cirkit.backend.torch.parameters.leaves import (
    TorchParameterLeaf,
    TorchParameterNode,
    TorchPointerParameter,
    TorchTensorParameter,
)


class TorchParameterOp(TorchParameterNode, ABC):
    def __init__(self, *in_shape: Tuple[int, ...], num_folds: int = 1):
        super().__init__(num_folds=num_folds)
        self.in_shapes = in_shape

    def __copy__(self) -> "TorchParameterOp":
        cls = self.__class__
        return cls(*self.in_shapes, **self.config)

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
    def from_leaf(cls, p: TorchParameterLeaf) -> "TorchParameter":
        return TorchParameter([p], {}, {}, topologically_ordered=True)

    @classmethod
    def from_sequence(
        cls, p: Union[TorchParameterLeaf, "TorchParameter"], *ns: TorchParameterNode
    ) -> "TorchParameter":
        if isinstance(p, TorchParameterLeaf):
            p = TorchParameter.from_leaf(p)
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
    def from_nary(
        cls, n: TorchParameterOp, *ps: Union[TorchParameterLeaf, "TorchParameter"]
    ) -> "TorchParameter":
        ps = tuple(
            TorchParameter.from_leaf(p) if isinstance(p, TorchParameterLeaf) else p for p in ps
        )
        p_nodes = list(chain.from_iterable(p.nodes for p in ps)) + [n]
        in_nodes = reduce(operator.ior, (p.nodes_inputs for p in ps), {})
        out_nodes = reduce(operator.ior, (p.nodes_outputs for p in ps), {})
        in_nodes[n] = list(p.output for p in ps)
        for p in ps:
            out_nodes[p.output] = [n]
        topologically_ordered = all(p.is_topologically_ordered for p in ps)
        return TorchParameter(
            p_nodes,
            in_nodes,
            out_nodes,
            topologically_ordered=topologically_ordered,
        )

    @classmethod
    def from_unary(
        cls, n: TorchUnaryOpParameter, p: Union[TorchParameterLeaf, "TorchParameter"]
    ) -> "TorchParameter":
        return TorchParameter.from_sequence(p, n)

    @classmethod
    def from_binary(
        cls,
        n: TorchBinaryOpParameter,
        p1: Union[TorchParameterLeaf, "TorchParameter"],
        p2: Union[TorchParameterLeaf, "TorchParameter"],
    ) -> "TorchParameter":
        return TorchParameter.from_nary(n, p1, p2)

    def extract_subgraphs(self, *roots: TorchParameterNode) -> List["TorchParameter"]:
        nodes_ptensor = set()

        def replace_ref_or_copy(n: TorchParameterNode) -> TorchParameterNode:
            if isinstance(n, TorchTensorParameter):
                if n in nodes_ptensor:
                    return TorchPointerParameter(n)
                nodes_ptensor.add(n)
            return shallowcopy(n)

        pgraphs = []
        for r in roots:
            nodes_map = {}
            in_nodes = {}
            out_nodes = defaultdict(list)
            for n in self.topological_ordering(roots=[r]):
                new_n = replace_ref_or_copy(n)
                nodes_map[n] = new_n
                in_new_nodes = [nodes_map[ni] for ni in self.node_inputs(n)]
                in_nodes[new_n] = in_new_nodes
                for ni in in_new_nodes:
                    out_nodes[ni].append(new_n)
            nodes = [nodes_map[n] for n in nodes_map.keys()]
            pgraph = TorchParameter(nodes, in_nodes, out_nodes, topologically_ordered=True)
            pgraphs.append(pgraph)
        return pgraphs

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
