import abc
import torch
import itertools
import einops as E
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

from torch import Tensor, nn

from cirkit.backend.torch.graph.address_book import AddressBook, FoldIndexInfo
from cirkit.utils.algorithms import DiAcyclicGraph
from cirkit.utils.scope import Scope


class AbstractTorchModule(nn.Module, ABC):
    def __init__(self, *, num_folds: int = 1):
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def fold_settings(self) -> Tuple[Any, ...]:
        ...


TorchModule = TypeVar("TorchModule", bound=AbstractTorchModule)


class TorchDiAcyclicGraph(nn.Module, DiAcyclicGraph[TorchModule], ABC):
    def __init__(
        self,
        modules: List[TorchModule],
        in_modules: Dict[TorchModule, List[TorchModule]],
        outputs: List[TorchModule],
        *,
        topologically_ordered: bool = False,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ):
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(nn.Module, self).__init__(
            modules, in_modules, outputs, topologically_ordered=topologically_ordered
        )
        self._address_book: Optional[AddressBook] = None
        self._fold_idx_info = fold_idx_info
        self._device = None

    @property
    def has_address_book(self) -> bool:
        return self._address_book is not None

    @property
    def device(self) -> Optional[Union[str, torch.device, int]]:
        return self._device

    def initialize_address_book(self) -> None:
        if self.has_address_book:
            raise RuntimeError("The address book has already been initialized")

        # Build the book address entries
        self._address_book = self._build_address_book()

    def to(
        self,
        device: Optional[Union[str, torch.device, int]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "TorchDiAcyclicGraph":
        if device is not None:
            self._address_book.set_device(device)
            self._device = device
        return cast(TorchDiAcyclicGraph, super().to(device, dtype, non_blocking))

    @abc.abstractmethod
    def _build_address_book(self) -> AddressBook:
        ...

    def _eval_forward(self, x: Optional[Tensor] = None) -> Tensor:
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        if not self.has_address_book:
            raise RuntimeError("The address book has not been initialized")
        module_outputs: List[Tensor] = []
        inputs_iterator = self._address_book.lookup(module_outputs, in_graph=x)
        for module, inputs in itertools.zip_longest(self.topological_ordering(), inputs_iterator):
            if module is None:
                (output,) = inputs
                return output
            y = module(*inputs)
            module_outputs.append(y)

    def _extended_eval_forward(self, x: Optional[Tensor], branches: List[Tensor]) -> Tensor:
        branches_queue = branches.copy()
        module_outputs: List[Tensor] = []
        inputs_iterator = self._address_book.lookup(module_outputs, in_graph=x)
        for module, inputs in itertools.zip_longest(self.topological_ordering(), inputs_iterator):
            if module is None:
                (output,) = inputs
                return output
            elif "Mixing" in str(type(module)) or "Dense" in str(type(module)):
                # elif isinstance(module, (TorchMixingLayer, TorchDenseLayer)):
                current_branch = branches_queue.pop(0)
                y = module.extended_forward(*inputs, current_branch)
            else:
                y = module.extended_forward(*inputs)
            module_outputs.append(y)

    def _pad_samples(self, y: Tensor, scope: Scope) -> Tensor:
        """
        Pads univariate samples to the size of the scope of the circuit (output dimension) according to scope for
        compatibility in downstream inner nodes. Currently only supports unfolded or folded where the leaf nodes
        are all folded into one fold.

        @param y: The samples to pad
        @shape y: (folds, channels, cardinality, samples)
        @param scope: The scope of the leaf node module outputting the samples

        @return: The padded samples
        @return shape: (folds, channels, cardinality, samples, circuit_scope)
        """
        # Ideally, here we need to pad with the zero element of the semiring, I think.
        # pad = torch.ones_like(y[..., 0]) * self.semiring.zero

        pad = torch.zeros_like(y)

        padded_samples = []
        if y.shape[0] == 1:
            running_scope = 0
            for variable in self.scope:
                if variable in scope:
                    padded_samples.append(y)
                    running_scope += 1
                else:
                    padded_samples.append(pad)
            padded_samples = torch.stack(padded_samples, dim=-1)
        elif len(self.scope) == y.shape[0]:
            c = y.shape[1]
            k = y.shape[2]

            padded_samples = E.repeat(y, "f c k n -> (c k n) f d", d=len(self.scope))
            indicator = torch.eye(
                len(self.scope), dtype=padded_samples.dtype, device=padded_samples.device
            )
            indicator = indicator.unsqueeze(0)

            padded_samples = padded_samples * indicator
            padded_samples = E.rearrange(padded_samples, "(c k n) f d -> f c k n d", c=c, k=k)
        else:
            # Arbitrary padding would require more information on the scope,
            # i.e. having a folded scope, otherwise not enough information.
            raise NotImplementedError("Padding for arbitrary folds not supported!")
        return padded_samples

    def _sample_forward(self, num_samples: int) -> Tuple[Tensor, List[Tensor]]:
        """
        Sample the computational graph by following the topological ordering forwards

        @param num_samples: The number of samples to generate

        @return: The samples generated by the computational graph
        @return shape: (folds, channels, cardinality, samples, circuit_scope)
        """

        module_outputs: List[Tensor] = []
        mixture_outputs: List[Tensor] = []
        inputs_iterator = self._address_book.lookup(module_outputs)
        for module, inputs in itertools.zip_longest(self.topological_ordering(), inputs_iterator):
            if module is None:
                (output,) = inputs
                return output, mixture_outputs
            elif inputs == ():
                # input nodes take no inputs for sampling
                y = module.sample_forward(num_samples)
                y = self._pad_samples(y, module.scope)
            else:
                # inner nodes take inputs for sampling
                y = module.sample_forward(num_samples, *inputs)
            if type(y) is tuple:
                module_outputs.append(y[0])
                mixture_outputs.append(y[1])
            else:
                module_outputs.append(y)
