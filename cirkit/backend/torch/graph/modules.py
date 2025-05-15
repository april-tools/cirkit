from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from torch import Tensor, nn

from cirkit.utils.algorithms import DiAcyclicGraph, subgraph


class AbstractTorchModule(nn.Module, ABC):
    """An abstract class representing a [torch.nn.Module][torch.nn.Module] that can be folded.

    An abstract torch module is used as base class for both the circuit layers and the nodes
    of the computational graph of the parameters of each layer.
    """

    def __init__(self, *, num_folds: int = 1):
        """Initialize the abstract torch module object.

        Args:
            num_folds: The number of folds computed by the module.
        """
        super().__init__()
        self._num_folds = num_folds

    @property
    def num_folds(self) -> int:
        """Retrieve the number of folds.

        Returns:
            The number of folds.
        """
        return self._num_folds

    @property
    @abstractmethod
    def fold_settings(self) -> tuple[Any, ...]:
        """Retrieve a tuple of attributes on which modules must agree on in order to be folded.

        Returns:
            A tuple of attributes.
        """

    @property
    def sub_modules(self) -> Mapping[str, "AbstractTorchModule"]:
        """Retrieve a dictionary mapping string identifiers to torch sub-modules,
        that must be passed to the ```__init__``` method of the top-level torch module.

        Returns:
            A dictionary of torch modules.
        """
        return {}


TorchModule = TypeVar("TorchModule", bound=AbstractTorchModule)
"""TypeVar: A torch module type that subclasses
    [AbstractTorchModule][cirkit.backend.torch.graph.modules.AbstractTorchModule]."""


@dataclass(frozen=True)
class FoldIndexInfo:
    """The folding index information of a folded computational graph, i.e.,
    a [directed acylic graph][cirkit.backend.torch.graph.modules.TorchDiAcyclicGraph]
    of [torch modules][cirkit.backend.torch.graph.modules.AbstractTorchModule].

    This data class stores (i) the topological ordering, (ii) the input fold index
    information for each torch module, and (iii) the output fold index information.
    """

    ordering: list[TorchModule]
    """The topological ordering of torch modules."""
    in_fold_idx: dict[int, list[list[tuple[int, int]]]]
    """The input fold index information. For each module index, it stores for each output
    fold computed by the module (first list), and for each input to the module (second list
    whose length is the arity), a tuple of (1) the input module index and (2) the fold index
    within that input module."""
    out_fold_idx: list[tuple[int, int]]
    """The output fold index information. For each output (first list), it stores a tuple of
    (1) the output module index and (2) the fold index within that output module."""


@dataclass(frozen=True)
class AddressBookEntry:
    """An entry of the address book data structure, which stores (i) the module (if it is
    not an output entry (i.e., an entry used to compute the output of the whole
    computational graph), and for each input module to it, (ii) it stores the unique indices
    of other modules, and (iii) the (optionally None) fold index tensor to apply in order
    to recover the input tensors to each fold.
    """

    module: TorchModule | None
    """The module the entry refers to. It can be None if the entry is then used to
    compute the output of the whole computational graph."""
    in_module_ids: list[list[int]]
    """For each input module, it stores the list of other module indices."""
    in_fold_idx: list[Tensor | tuple[slice | None, ...]]
    """For each input module, it stores the fold index tensor used to gather the
    input tensors to each fold. It is a tuple of optional slices whether there is no need of
    gathering the input tensors, i.e., if the indexing operation would act as an unsqueezing
    operation that can be much more efficient."""


class AddressBook(nn.Module, ABC):
    """The address book data structure, sometimes also known as book-keeping.
    The address book stores a list of
    [AddressBookEntry][cirkit.backend.torch.graph.modules.AddressBookEntry],
    where each entry stores the information needed to gather the inputs to each (possibly folded)
    torch module.
    """

    def __init__(self, entries: list[AddressBookEntry]) -> None:
        """Initializes an address book.

        Args:
            entries: The list of address book entries.

        Raises:
            ValueError: If the list of address book entries is empty.
            ValueError: If the last entry (i.e., the entry used to compute the output of
                the whole computational graph) has a torch module assigned to it, or
                if it has more than one fold index tensor, or if the fold index tensor
                is not a 1-dimensional tensor.
        """
        if not entries:
            raise ValueError("The list of address book entry must not be empty")
        last_entry = entries[-1]
        if last_entry.module is not None:
            raise ValueError(
                "The last entry of the address book must not have a module associated to it"
            )
        if len(last_entry.in_fold_idx) != 1:
            raise ValueError(
                "The last entry of the address book must have only one fold index tensor"
            )
        (out_fold_idx,) = last_entry.in_fold_idx
        if not isinstance(out_fold_idx, Tensor) or len(out_fold_idx.shape) != 1:
            raise ValueError("The output fold index tensor should be a 1-dimensional tensor")
        super().__init__()
        self._num_outputs = out_fold_idx.shape[0]
        self._entry_modules: list[TorchModule | None] = [e.module for e in entries]
        self._entry_in_module_ids: list[list[list[int]]] = [e.in_module_ids for e in entries]
        # We register the book-keeping tensor indices as buffers.
        # By doing so they are automatically transferred to the device
        # This reduces CPU-device communications required to transfer these indices
        #
        # TODO: Perhaps this can be made more elegant in the future, if someone
        #  decides to introduce a nn.BufferList container in torch
        self._entry_in_fold_idx_targets: list[list[str]] = []
        for i, e in enumerate(entries):
            self._entry_in_fold_idx_targets.append([])
            for j, fi in enumerate(e.in_fold_idx):
                in_fold_idx_target = f"_in_fold_idx_{i}_{j}"
                if isinstance(fi, Tensor):
                    self.register_buffer(in_fold_idx_target, fi)
                else:
                    setattr(self, in_fold_idx_target, fi)
                self._entry_in_fold_idx_targets[-1].append(in_fold_idx_target)

    def __len__(self) -> int:
        """Retrieve the length of the address book.

        Returns:
            The number of address book entries.
        """
        return len(self._entry_modules)

    def __iter__(self) -> Iterator[AddressBookEntry]:
        """Retrieve an iterator over address book entries, i.e., a tuple consisting of
        three objects: (i) the torch module to evaluate (it can be None if the entry
        is needed to return the output of the computational graph); (ii) for each input
        to the module (i.e., depending on the arity) we have the list of ids to the
        outputs of other modules (it can be empty if the module is an input module); and
        (iii) for each input to the module we have the fold indexing, which
        is used to retrieve the inputs to a module, even if they are folded modules.

        Returns:
            An iterator over address book entries.
        """
        for module, in_module_ids_hs, in_fold_idx_targets in zip(
            self._entry_modules, self._entry_in_module_ids, self._entry_in_fold_idx_targets
        ):
            yield AddressBookEntry(
                module,
                in_module_ids_hs,
                [getattr(self, target) for target in in_fold_idx_targets],
            )

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the whole computational graph represented
        through the address book.

        For instance, for a circuit with $n$ output layers, this will be equal to $n$.

        Returns:
            The number of outputs.
        """
        return self._num_outputs

    @abstractmethod
    def lookup(
        self, module_outputs: list[Tensor], *, in_graph: Tensor | None = None
    ) -> Iterator[tuple[TorchModule | None, tuple]]:
        """Retrive an iterator that iteratively returns a torch module and the tensor inputs to it.

        Args:
            module_outputs: A list of the outputs of each torch module. This list is expected to
                be iteratively expanded as we continue evaluating the modules of the torch
                computational graph.
            in_graph: An optional tensor input to the whole computational graph. This is used
                as input to the torch modules that do not receive input from other torch
                modules within the torch computationa graph.

        Returns:
            An iterator of tuples, where the first element is a torch module if we are
            retriving the inputs to it, and None if we are retrieving the output of the
            whole computational graph (i.e., in the final step of the evaluation).
            The second element is instead a tuple of arguments that are input to the
            torch module (e.g., some input tensors)
        """


class ModuleEvalFunction(Protocol):  # pylint: disable=too-few-public-methods
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
        modules: Sequence[TorchModule],
        in_modules: Mapping[TorchModule, Sequence[TorchModule]],
        outputs: Sequence[TorchModule],
        *,
        fold_idx_info: FoldIndexInfo | None = None,
    ):
        """Initialize a torch computational graph.

        Args:
            modules: The module nodes.
            in_modules: A dictionary mapping modules to their input modules, if any.
            outputs: A list of modules that are the output modules in the computational graph.
            fold_idx_info: The folding index information.
                It can be None if the Torch graph is not folded.
        """
        modules: list[TorchModule] = nn.ModuleList(modules)  # type: ignore
        super().__init__()
        super(nn.Module, self).__init__(modules, in_modules, outputs)
        self._is_folded = fold_idx_info is not None
        if fold_idx_info is None:
            fold_idx_info = self._build_unfold_index_info()
        self._address_book = self._build_address_book(fold_idx_info)

    @property
    def is_folded(self) -> bool:
        """Retrieves whether the computational graph is folded or not.

        Returns:
            True if it is folded, False otherwise.
        """
        return self._is_folded

    @property
    def address_book(self) -> AddressBook:
        """Retrieve the address book object of the computational graph.

        Returns:
            The address book.
        """
        return self._address_book

    def subgraph(self, *roots: TorchModule) -> "TorchDiAcyclicGraph[TorchModule]":
        """Assuming the computational graph is not a folded one,
        this returns the sub-graph having the given root torch modules as output modules.

        Args:
            *roots: The root torch modules of the sub-graph to return.

        Returns:
            A new torch computational graph having the given roots as the output torch modules.

        Raises:
            ValueError: If the computational graph is folded.
        """
        if self.is_folded:
            raise ValueError("Cannot extract a sub-computational graph from a folded one")
        nodes, in_nodes = subgraph(roots, self.node_inputs)
        return self.__class__(nodes, in_nodes, outputs=roots)

    def evaluate(
        self, x: Tensor | None = None, module_fn: ModuleEvalFunction | None = None
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

        Raises:
            RuntimeError: If the address book is somehow not well-formed.
        """
        # Evaluate the computational graph by following the topological ordering,
        # and by using the book address information to retrieve the inputs to each
        # (possibly folded) torch module.
        module_outputs: list[Tensor] = []
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
