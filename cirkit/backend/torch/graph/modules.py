import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph.address_book import AddressBook, FoldIndexInfo
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


class TorchDiAcyclicGraph(nn.Module, DiAcyclicGraph[TorchModule], ABC):
    """A torch directed acyclic graph module, i.e., a computational graph made of torch modules."""

    def __init__(
        self,
        modules: List[TorchModule],
        in_modules: Dict[TorchModule, List[TorchModule]],
        outputs: List[TorchModule],
        *,
        topologically_ordered: bool = False,
        fold_idx_info: Optional[FoldIndexInfo] = None,
    ):
        """Initialize a Torch computational graph.

        Args:
            modules: The module nodes.
            in_modules: A dictionary mapping modules to their input modules, if any.
            outputs: A list of modules that are the output modules in the computational graph.
            topologically_ordered: A flag indicating if the given module nodes list is already
                topologically ordered. If set to True then it can speed up evaluation.
            fold_idx_info: The folding index information. It can be None if the Torch graph is
                not folded. This will be consumed (i.e., set to None) when the address book data
                structure is built.
        """
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
        """Check whether the Torch graph has already an address book.

        Returns:
            True if the address book has been initialized, and False otherwise.
        """
        return self._address_book is not None

    @property
    def device(self) -> Optional[Union[str, torch.device, int]]:
        """Retrieve the device the module is allocated to.

        Returns:
            A device, which can be a string, and integer or a torch.device object.
        """
        return self._device

    def initialize_address_book(self) -> None:
        """Initialize the address book data structure. This can only be called once, and consumes
            the folding index information, if any.

        Raises:
            RuntimeError: Raised if the address book has not been initialized.
        """
        if self.has_address_book:
            raise RuntimeError("The address book has already been initialized")
        # Build the book address entries
        self._address_book = self._build_address_book()
        self._fold_idx_info = None

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

    @abstractmethod
    def _build_address_book(self) -> AddressBook:
        """Construct the address book data structure. This should be implemented by any subclass.

        Returns:
            The address book data structure.
        """

    def _eval_forward(self, x: Optional[Tensor] = None) -> Tensor:
        """Evaluate the Torch graph by following the topological ordering,
            and by using the address book information to retrieve the inputs to each module.

        Args:
            x: The input of the Torch graph. It can be None.

        Returns:
            The output tensor of the Torch graph.
            If the Torch graph has multiple outputs, then they will be stacked.
        """
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
        raise RuntimeError("The address book is malformed")
