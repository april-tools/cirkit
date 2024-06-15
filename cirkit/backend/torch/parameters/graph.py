from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

from torch import Tensor, nn

from cirkit.backend.torch.graph import TorchRootedDiAcyclicGraph, AddressBook, AbstractAddressBook


class TorchParameterNode(nn.Module, ABC):
    """The abstract base class for all reparameterizations."""

    def __init__(self, *, num_folds: int = 1, **kwargs) -> None:
        """Init class."""
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return {}


class TorchParameterLeaf(TorchParameterNode, ABC):
    @property
    def is_initialized(self) -> bool:
        return True

    def initialize_(self) -> None:
        pass

    def __call__(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        ...


class TorchParameterOp(TorchParameterNode, ABC):
    def __init__(self, *in_shape: Tuple[int, ...], num_folds: int = 1):
        super().__init__(num_folds=num_folds)
        self.in_shapes = in_shape

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


class TorchParameter(TorchRootedDiAcyclicGraph[TorchParameterNode]):
    def __init__(
        self,
        nodes: List[TorchParameterNode],
        in_nodes: Dict[TorchParameterNode, List[TorchParameterNode]],
        out_nodes: Dict[TorchParameterNode, List[TorchParameterNode]],
        *,
        topologically_ordered: bool = False,
        address_book: Optional[AbstractAddressBook] = None,
    ) -> None:
        if address_book is None:
            address_book = AddressBook()
        super().__init__(
            nodes, in_nodes, out_nodes,
            topologically_ordered=topologically_ordered,
            address_book=address_book
        )

    @property
    def num_folds(self) -> int:
        return self.output.num_folds

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.output.shape

    def initialize_(self) -> None:
        """Reset the input parameters."""
        # TODO: assuming parameter operators do not have any learnable parameters
        for p in self.inputs:
            p.initialize_()

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def forward(self) -> Tensor:
        y = self.eval_forward()  # (F, d1, d2, ..., dk)
        return y
