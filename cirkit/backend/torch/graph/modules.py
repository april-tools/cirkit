import abc
from abc import ABC
from functools import cached_property
from typing import Dict, List, Optional

from torch import Tensor, nn

from cirkit.backend.torch.graph.folding import AddressBook, FoldIndexInfo, TorchModuleType
from cirkit.utils.algorithms import DiAcyclicGraph


class TorchDiAcyclicGraph(nn.Module, ABC, DiAcyclicGraph[TorchModuleType]):
    def __init__(
        self,
        modules: List[TorchModuleType],
        in_modules: Dict[TorchModuleType, List[TorchModuleType]],
        out_modules: Dict[TorchModuleType, List[TorchModuleType]],
        *,
        topologically_ordered: bool = False,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(ABC, self).__init__(
            modules, in_modules, out_modules, topologically_ordered=topologically_ordered
        )
        self._address_book: Optional[AddressBook] = None
        self._fold_idx_info = fold_idx_info

    @property
    def has_address_book(self) -> bool:
        return self._address_book is not None

    def initialize_address_book(self) -> None:
        if self.has_address_book:
            raise RuntimeError("The address book has already been initialized")
        self._address_book = self._build_address_book()

    @abc.abstractmethod
    def _build_address_book(self) -> AddressBook:
        ...

    def _eval_forward(self, x: Optional[Tensor] = None) -> Tensor:
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        module_outputs: List[Tensor] = []
        lookup_iterator = self._address_book.lookup(module_outputs, in_graph=x)
        for module in self.topological_ordering():
            inputs = next(lookup_iterator)
            y = module(*inputs)
            module_outputs.append(y)

        # Retrieve the output tensor from the address book
        (output,) = next(lookup_iterator)
        return output


class TorchRootedDiAcyclicGraph(TorchDiAcyclicGraph[TorchModuleType]):
    @cached_property
    def output(self) -> TorchModuleType:
        outputs = list(self.outputs)
        if len(outputs) > 1:
            raise ValueError("The graph has more than one output node.")
        return outputs[0]
