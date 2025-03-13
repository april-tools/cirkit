from collections.abc import Iterator, Sequence

import torch
from torch import Tensor

from cirkit.backend.torch.graph.folding import (
    build_address_book_entry,
    build_address_book_stacked_entry,
    build_unfold_index_info,
)
from cirkit.backend.torch.graph.modules import (
    AddressBook,
    AddressBookEntry,
    FoldIndexInfo,
    TorchDiAcyclicGraph,
)
from cirkit.backend.torch.parameters.nodes import TorchParameterNode

from cirkit.utils.shape import comp_shape


class ParameterAddressBook(AddressBook):
    """The address book data structure for the parameter computational graphs.
    See [TorchParameter][cirkit.backend.torch.parameters.parameter.TorchParameter].
    The address book stores a list of
    [AddressBookEntry][cirkit.backend.torch.graph.modules.AddressBookEntry],
    where each entry stores the information needed to gather the inputs to each (possibly folded)
    node in the parameter computational graph.
    """

    def lookup(
        self, module_outputs: list[Tensor], *, in_graph: Tensor | None = None
    ) -> Iterator[tuple[TorchParameterNode | None, tuple]]:
        def _select_index(mids: list[int], idx: Tensor | tuple[slice | None, ...]) -> Tensor:
            # A useful function combining the modules outputs, and then possibly applying an index
            if len(mids) == 1:
                t = module_outputs[mids[0]]
            else:
                t = torch.cat([module_outputs[mid] for mid in mids], dim=0)
            return t[idx]

        # Loop through the entries and yield inputs
        for entry in self:
            node = entry.module
            in_node_ids = entry.in_module_ids
            in_fold_idx = entry.in_fold_idx
            # Catch the case there are some inputs coming from other modules
            if in_node_ids:
                x = tuple(
                    _select_index(mids, in_idx) for mids, in_idx in zip(in_node_ids, in_fold_idx)
                )
                yield node, x
                continue

            # Catch the case there are no inputs coming from other modules
            yield node, ()

    @classmethod
    def from_index_info(cls, fold_idx_info: FoldIndexInfo) -> "ParameterAddressBook":
        """Constructs the parameter nodes address book using fold index information.

        Args:
            fold_idx_info: The fold index information.

        Returns:
            A parameter nodes address book.
        """
        # The address book entries being built
        entries: list[AddressBookEntry] = []

        # A useful dictionary mapping module ids to their number of folds
        num_folds: dict[int, int] = {}

        # Build the bookkeeping data structure by following the topological ordering
        for mid, m in enumerate(fold_idx_info.ordering):
            # Retrieve the index information of the input modules
            in_modules_fold_idx = fold_idx_info.in_fold_idx[mid]

            # Catch the case of a folded module having the input of the network as input
            if in_modules_fold_idx:
                entry = build_address_book_entry(m, in_modules_fold_idx, num_folds=num_folds)
            # Catch the case of a folded module without inputs
            else:
                entry = AddressBookEntry(m, [], [])

            num_folds[mid] = m.num_folds
            entries.append(entry)

        # Append the last bookkeeping entry with the information to compute the output tensor
        entry = build_address_book_stacked_entry(
            None, [fold_idx_info.out_fold_idx], num_folds=num_folds, output=True
        )
        entries.append(entry)

        return ParameterAddressBook(entries)


class TorchParameter(TorchDiAcyclicGraph[TorchParameterNode]):
    r"""A torch parameter is a computational graph consisting of computational nodes,
    and computing a tensor parameter that is then used by a circuit layer. That is,
    given F the number of folds, and (K_1,\ldots,K_n) the shape of each parameter fold, a
    torch parameter computes a tensor of shape (F,K_1,\ldots,K_n).
    Note that a torch parameter does not take any tensor as input.
    """

    def __init__(
        self,
        modules: Sequence[TorchParameterNode],
        in_modules: dict[TorchParameterNode, Sequence[TorchParameterNode]],
        outputs: Sequence[TorchParameterNode],
        *,
        fold_idx_info: FoldIndexInfo | None = None,
    ):
        """Initialize a torch parameter computational graph.

        Args:
            modules: The parameter computational nodes.
            in_modules: A dictionary mapping nodes to their input nodes, if any.
            outputs: A list of nodes that are the output nodes in the computational graph.
            fold_idx_info: The folding index information.
                It can be None if the Torch graph is not folded.
        """
        super().__init__(modules, in_modules, outputs, fold_idx_info=fold_idx_info)

    @property
    def device(self) -> torch.device:
        """Retrieve the device of the parameter computational graph.
        Currently, it assumes all [torch.nn.parameter.Parameter][torch.nn.parameter.Parameter]
        it contains are stored in the same device.

        Returns:
            torch.device: The device.
        """
        # TODO: Obtaining the device in this way is only needed because
        #  the integrate() method in layers can output constant tensors that
        #  would be allocated on the CPU by default (e.g., log_partition_function()).
        #  Since having a device flag in nn.Module is malpractice (tensors are stored
        #  on devices but NOT modules), is there a better way to do this?
        return next(self.parameters()).device

    @property
    def num_folds(self) -> int:
        """The number of folds of the computed tensor parameter.

        Returns:
            The number of folds.
        """
        return self._address_book.num_outputs

    @property
    def shape(self) -> tuple[int, ...]:
        r"""The shape of the computed tensor parameter, without considering
        the number of folds. That is, if the number of folds
        (see [TorchParameter.num_folds][cirkit.backend.torch.parameters.parameter.TorchParameter.num_folds])
        is F and the shape is (K_1,\ldots,K_n), it means the torch parameter
        computes a tensor of shape (F, B, K_1,\ldots,K_n) where B is the batch size.
        A batched parameter means that each sample in an input is parameterized by a different tensor.
        If the parameter has a batch size of 1 then it is broadcasted to all the samples in the input
        batch.

        Returns:
            The shape of the computed tensor parameter, without considering the number of folds.
            The value -1 is used to indicate an unknown batch size.
        """
        return (-1, *self.outputs[0].shape)

    def reset_parameters(self) -> None:
        """Reset the parameters of the parameter computational graph."""
        for p in self.nodes:
            p.reset_parameters()

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        r"""Evaluate the parameter computational graph.

        Returns:
            Tensor: The output parameter tensor, having shape (F, B, K_1,\ldots K_n),
                where F is the number of folds, B the batch size and (K_1,\ldots,K_n) 
                is the shape of each parameter tensor slice.
        """
        return self.evaluate()

    def _build_unfold_index_info(self) -> FoldIndexInfo:
        return build_unfold_index_info(
            self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
        )

    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        return ParameterAddressBook.from_index_info(fold_idx_info)

    def extra_repr(self) -> str:
        return f"shape: {(self.num_folds, *self.shape)}"
