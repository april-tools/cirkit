from abc import ABC
from functools import cached_property
from typing import Dict, List, Optional, TypeVar

from torch import nn, Tensor

from cirkit.backend.torch.graph.books import AbstractAddressBook
from cirkit.backend.torch.graph.nodes import TorchModule
from cirkit.utils.algorithms import DiAcyclicGraph


TorchModuleType = TypeVar('TorchModuleType', bound=TorchModule)


class TorchDiAcyclicGraph(nn.Module, ABC, DiAcyclicGraph[TorchModuleType]):
    def __init__(
        self,
        modules: List[TorchModuleType],
        in_modules: Dict[TorchModuleType, List[TorchModuleType]],
        out_modules: Dict[TorchModuleType, List[TorchModuleType]],
        *,
        topologically_ordered: bool = False,
        address_book: AbstractAddressBook,
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(ABC, self).__init__(modules, in_modules, out_modules, topologically_ordered=topologically_ordered)
        self._address_book = address_book

    def initialize_address_book(self) -> None:
        if self._address_book.is_built:
            raise ValueError("The address book has already been built")
        self._address_book.build(
            self.topological_ordering(),
            outputs=self.outputs,
            incomings_fn=self.node_inputs
        )

    def _in_index(self, x: Tensor, idx: Tensor) -> Tensor:
        raise NotImplementedError()

    def _eval_forward(self, x: Optional[Tensor] = None) -> Tensor:
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        def _in_network_fn(idx: Tensor) -> Tensor:
            return self._in_index(x, idx)
        in_network_fn = _in_network_fn if x is not None else None
        module_outputs: List[Tensor] = []
        for i, module in enumerate(self.topological_ordering()):
            inputs = self._address_book.retrieve_module_inputs(
                i, module_outputs, in_network_fn=in_network_fn
            )
            y = module(*inputs)
            module_outputs.append(y)

        # Retrieve the output from the book address
        return self._address_book.retrieve_output(module_outputs)


class TorchRootedDiAcyclicGraph(TorchDiAcyclicGraph[TorchModuleType]):
    @cached_property
    def output(self) -> TorchModuleType:
        outputs = list(self.outputs)
        if len(outputs) > 1:
            raise ValueError("The graph has more than one output node.")
        return outputs[0]
