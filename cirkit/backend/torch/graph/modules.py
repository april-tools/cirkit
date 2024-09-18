from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, TypeVar, Union, cast

import torch
from torch import Tensor, nn

from cirkit.utils.algorithms import DiAcyclicGraph


class AbstractTorchModule(nn.Module, ABC):
    """An abstract class representing a torch.nn.Module that can be folded."""

    def __init__(self, *, num_folds: int = 1):
        """Initialize the abstract torch module object.

        Args:
            num_folds: The number of folds computed by the module.
        """
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def fold_settings(self) -> Tuple[Any, ...]:
        """Retrieve a tuple of attributes on which modules must agree on in order to be folded.

        Returns:
            A tuple of attributes.
        """


TorchModule = TypeVar("TorchModule", bound=AbstractTorchModule)
"""TypeVar: A torch module type that subclasses
    [AbstractTorchModule][cirkit.backend.torch.graph.modules.AbstractTorchModule]."""


@dataclass(frozen=True)
class FoldIndexInfo:
    ordering: List[TorchModule]
    in_fold_idx: Dict[int, List[List[Tuple[int, int]]]]
    out_fold_idx: List[Tuple[int, int]]


@dataclass(frozen=True)
class AddressBookEntry:
    module: Optional[TorchModule]
    in_module_ids: List[List[int]]
    in_fold_idx: List[Optional[Tensor]]


class AddressBook(ABC):
    def __init__(self, entries: List[AddressBookEntry]) -> None:
        super().__init__()
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[AddressBookEntry]:
        return iter(self._entries)

    def set_device(self, device: Optional[Union[str, torch.device, int]] = None) -> "AddressBook":
        self._entries = list(
            map(
                lambda entry: AddressBookEntry(
                    entry.module,
                    entry.in_module_ids,
                    [idx if idx is None else idx.to(device) for idx in entry.in_fold_idx],
                ),
                self._entries,
            )
        )
        return self

    @abstractmethod
    def lookup(
        self, module_outputs: List[Tensor], *, in_graph: Optional[Tensor] = None
    ) -> Iterator[Tuple[Optional[TorchModule], Tuple[Tensor, ...]]]:
        ...


class ModuleEvalFunctional(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a function that evaluates a module on some inputs."""

    def __call__(self, module: TorchModule, *inputs: Tensor) -> Tensor:
        """Evaluate a module on some inputs.

        Args:
            module: The module to evaluate.
            inputs: The tensor inputs to the module

        Returns:
            Tensor: The output of the module as specified by this functional.
        """


class TorchDiAcyclicGraph(nn.Module, DiAcyclicGraph[TorchModule], ABC):
    """A torch directed acyclic graph module, i.e., a computational graph made of torch modules."""

    def __init__(
        self,
        modules: List[TorchModule],
        in_modules: Dict[TorchModule, List[TorchModule]],
        outputs: List[TorchModule],
        *,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ):
        """Initialize a Torch computational graph.

        Args:
            modules: The module nodes.
            in_modules: A dictionary mapping modules to their input modules, if any.
            outputs: A list of modules that are the output modules in the computational graph.
            fold_idx_info: The folding index information. It can be None if the Torch graph is
                not folded. This will be consumed (i.e., set to None) when the address book data
                structure is built.
        """
        modules: List = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(nn.Module, self).__init__(modules, in_modules, outputs)
        self._device = None
        if fold_idx_info is None:
            fold_idx_info = self._build_unfold_index_info()
        self._address_book = self._build_address_book(fold_idx_info)

    @property
    def device(self) -> Optional[Union[str, torch.device, int]]:
        """Retrieve the device the module is allocated to.

        Returns:
            A device, which can be a string, and integer or a torch.device object.
        """
        return self._device

    @property
    def address_book(self) -> AddressBook:
        """Retrieve the address book object of the computational graph.

        Returns:
            The address book.
        """
        return self._address_book

    def to(
        self,
        device: Optional[Union[str, torch.device, int]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "TorchDiAcyclicGraph":
        """Specialization of the torch module's to() method. This is used to set the device
            attribute.

        Args:
            device: The device.
            dtype: The dtype.
            non_blocking: Whether the method should be non-blocking.

        Returns:
            Itself.
        """
        if device is not None:
            self._address_book.set_device(device)
            self._device = device
        return cast(TorchDiAcyclicGraph, super().to(device, dtype, non_blocking))

    def evaluate(
        self, x: Optional[Tensor] = None, module_fn: Optional[ModuleEvalFunctional] = None
    ) -> Tensor:
        """Evaluate the Torch graph by following the topological ordering,
            and by using the address book information to retrieve the inputs to each module.

        Args:
            x: The input of the Torch computational graph. It can be None.
            module_fn: A functional over modules that overrides the forward method defined by a
                module. It can be None. If it is None, then the ```__call__``` method defined by
                the module itself is used.

        Returns:
            The output tensor of the Torch graph.
            If the Torch graph has multiple outputs, then they will be stacked.
        """
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        module_outputs: List[Tensor] = []
        for module, inputs in self._address_book.lookup(module_outputs, in_graph=x):
            if module is None:
                (output,) = inputs
                return output
            if module_fn is None:
                y = module(*inputs)
            else:
                y = module_fn(module, *inputs)
            module_outputs.append(y)
        raise RuntimeError("The address book is malformed")

    @abstractmethod
    def _build_unfold_index_info(self) -> FoldIndexInfo:
        ...

    @abstractmethod
    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        ...

    def __repr__(self) -> str:
        def indent(s: str) -> str:
            s = s.split("\n")
            r = s[0]
            if len(s) == 1:
                return r
            return r + "\n" + "\n".join(f"  {t}" for t in s[1:])

        lines = [self.__class__.__name__ + "("]
        extra_lines = self.extra_repr()
        if extra_lines:
            lines.append(f"  {indent(extra_lines)}")
        for i, entry in enumerate(self._address_book):
            if entry.module is None:
                continue
            repr_module = indent(repr(entry.module))
            lines.append(f"  ({i}): {repr_module}")
        lines.append(")")
        return "\n".join(lines)
