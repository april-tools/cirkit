from collections import ChainMap
from collections.abc import Iterator, Sequence
from itertools import chain
from typing import Union

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
from cirkit.backend.torch.parameters.nodes import (
    TorchBinaryParameterOp,
    TorchParameterInput,
    TorchParameterNode,
    TorchParameterOp,
    TorchUnaryParameterOp,
)


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
        computes a tensor of shape (F, K_1,\ldots,K_n).

        Returns:
            The shape of the computed tensor parameter, without considering the number of folds.
        """
        return self.outputs[0].shape

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
            Tensor: The output parameter tensor, having shape (F, K_1,\ldots K_n),
                where F is the number of folds, and (K_1,\ldots,K_n) is the shape
                of each parameter tensor slice.
        """
        output = self.evaluate()[-1]
        return output

    def _build_unfold_index_info(self) -> FoldIndexInfo:
        return build_unfold_index_info(
            self.topological_ordering(), outputs=self.outputs, incomings_fn=self.node_inputs
        )

    def _build_address_book(self, fold_idx_info: FoldIndexInfo) -> AddressBook:
        return ParameterAddressBook.from_index_info(fold_idx_info)

    def extra_repr(self) -> str:
        return f"shape: {(self.num_folds, *self.shape)}"

    @classmethod
    def from_input(cls, p: TorchParameterInput) -> "TorchParameter":
        """Constructs a parameter from a leaf symbolic node only.

        Args:
            p: The parameter input.

        Returns:
            A parameter containing only the given parameter input.
        """
        return TorchParameter([p], {}, outputs=[p])

    @classmethod
    def from_sequence(
        cls, p: Union[TorchParameterInput, "TorchParameter"], *ns: TorchParameterOp
    ) -> "TorchParameter":
        """Constructs a parameter from a composition of parameter nodes.

        Args:
            p: The entry point of the sequence, which can be either a parameter
                input or another parameter.
            *ns: A sequence of parameter nodes.

        Returns:
            A parameter that encodes the composition of the parameter nodes,
                starting from the given entry point of the sequence.
        """
        assert p.num_folds == 1
        if isinstance(p, TorchParameterInput):
            p = TorchParameter.from_input(p)
        nodes = list(p.nodes) + list(ns)
        in_nodes = dict(p.nodes_inputs)
        for i, n in enumerate(ns):
            in_nodes[n] = [ns[i - 1]] if i - 1 >= 0 else [p.outputs[0]]
        return TorchParameter(nodes, in_nodes, [ns[-1]])

    @classmethod
    def from_nary(
        cls, n: TorchParameterOp, *ps: Union[TorchParameterInput, "TorchParameter"]
    ) -> "TorchParameter":
        """Constructs a parameter by using a parameter operation node and by specifying its inputs.

        Args:
            n: The parameter operation node.
            *ps: A sequence of parameter input nodes or parameters.

        Returns:
            A parameter that encodes the application of the given parameter operation node
                to the outputs given by the parameter input nodes or parameters.
        """
        assert n.num_folds == 1
        ps = tuple(
            TorchParameter.from_input(p) if isinstance(p, TorchParameterInput) else p for p in ps
        )
        assert all(len(p.outputs) == 1 for p in ps)
        p_nodes = list(chain.from_iterable(p.nodes for p in ps)) + [n]
        in_nodes = dict(ChainMap(*(p.nodes_inputs for p in ps)))
        in_nodes[n] = list(p.outputs[0] for p in ps)
        return TorchParameter(p_nodes, in_nodes, [n])

    @classmethod
    def from_unary(
        cls, n: TorchUnaryParameterOp, p: Union[TorchParameterInput, "TorchParameter"]
    ) -> "TorchParameter":
        """Constructs a parameter by using a unary parameter operation node and by specifying its
        inputs.

        Args:
            n: The unary parameter operation node.
            p: The parameter input node, or another parameter.

        Returns:
            A parameter that encodes the application of the given parameter operation
                node to the output given by the parameter input node or parameter.
        """
        assert n.num_folds == 1 and p.num_folds == 1
        return TorchParameter.from_sequence(p, n)

    @classmethod
    def from_binary(
        cls,
        n: TorchBinaryParameterOp,
        p1: Union[TorchParameterInput, "TorchParameter"],
        p2: Union[TorchParameterInput, "TorchParameter"],
    ) -> "TorchParameter":
        """Constructs a parameter by using a binary parameter operation node and by specifying
        its inputs.

        Args:
            n: The binary parameter operation node.
            p1: The first parameter input node, or another parameter.
            p2: The second parameter input node, or another parameter.

        Returns:
            A parameter that encodes the application of the given parameter operation
                node to the two outputs given by the parameter inputs or parameters.
        """
        assert n.num_folds == 1 and p1.num_folds == 1 and p2.num_folds == 1
        return TorchParameter.from_nary(n, p1, p2)
