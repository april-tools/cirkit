import abc
import torch
import itertools
from abc import ABC
from functools import cached_property
from typing import Dict, List, Optional

from torch import Tensor, nn

from cirkit.backend.torch.graph.folding import AddressBook, FoldIndexInfo
from cirkit.backend.torch.graph.nodes import TorchModuleType
from cirkit.utils.algorithms import DiAcyclicGraph


class TorchDiAcyclicGraph(nn.Module, DiAcyclicGraph[TorchModuleType], ABC):
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
        super(nn.Module, self).__init__(
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
        inputs_iterator = self._address_book.lookup(module_outputs, in_graph=x)
        for module, inputs in itertools.zip_longest(self.topological_ordering(), inputs_iterator):
            if module is None:
                (output,) = inputs
                return output
            y = module(*inputs)
            module_outputs.append(y)

    def _sample_forward(self, num_samples: int) -> Tensor:
        if self._fold_idx_info is not None:
            raise NotImplementedError("Folded forward sampling is not implemented yet!")

        """
        Sample the computational graph by following the topological ordering forwards
        """

        module_outputs: List[Tensor] = []
        mixture_outputs: List[Tensor] = []
        inputs_iterator = self._address_book.lookup(module_outputs)
        for module, inputs in itertools.zip_longest(self.topological_ordering(), inputs_iterator):
            if module is None:
                (output,) = inputs
                return output
            elif inputs == ():
                # input nodes take no inputs for sampling
                y = module.sample_forward(num_samples)
            else:
                # inner nodes take inputs for sampling
                y = module.sample_forward(num_samples, *inputs)

            if type(y) is tuple:
                module_outputs.append(y[0])
                mixture_outputs.append(y[1])
            else:
                module_outputs.append(y)

    def _sample_backward(self, num_samples: int) -> Tensor:
        if self._fold_idx_info is not None:
            raise NotImplementedError("Folded backward sampling is not implemented yet!")
        raise NotImplementedError("Backward sampling is not implemented yet!")

        """
        Sample the computational graph by following the topological ordering backwards
        """

        sample_dict = {layer: [] for layer in self.modules()}
        unit_dict = {layer: [] for layer in self.modules()}

        inner_module_weights: List[Tensor] = []
        input_modules: List[nn.Module] = []
        module_iterator = self.topological_ordering()
        for module in reversed(module_iterator):
            if module.num_input_units == 0:
                input_modules.append(module)
            else:
                sample_weights = module._sample_backward(
                    sample_dict=sample_dict,
                    unit_dict=unit_dict,
                    num_samples=num_samples,
                )
                inner_module_weights.append(sample_weights)

        samples = []
        for module in input_modules:
            samples.append(
                module.sample_backward(
                    sample_dict=sample_dict,
                    unit_dict=unit_dict,
                    num_samples=num_samples,
                )
            )

        samples = torch.stack(samples, dim=0)
        return samples


class TorchRootedDiAcyclicGraph(TorchDiAcyclicGraph[TorchModuleType], ABC):
    @cached_property
    def output(self) -> TorchModuleType:
        outputs = list(self.outputs)
        if len(outputs) != 1:
            raise ValueError("The graph has more than one output node.")
        (output,) = outputs
        return output
