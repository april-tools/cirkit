import functools
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
from typing_extensions import ParamSpec  # TODO: in typing from 3.10

import numpy as np
import torch
from torch import Tensor

# TODO: build a unified reproducibility handler


T = TypeVar("T")
P = ParamSpec("P")


class RandomCtx:
    """A context manager and a function decorator to handle the RNG state on a piece of code."""

    def __init__(self, seed: int) -> None:
        """Initialize the random state manager.

        Args:
            seed (int): The seed to seed all RNGs.
        """
        self.seed = seed % 2**32
        self.use_cuda: bool = torch.cuda._is_in_bad_fork()

        self.random_state: Tuple[Any, ...] = ()  # type: ignore[misc]
        self.numpy_state: Dict[str, Any] = {}  # type: ignore[misc]
        self.torch_state = Tensor()
        self.cuda_state: List[Tensor] = []

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Wrap a function to run under the random context.

        Args:
            func (Callable[P, T]): The function to wrap.

        Returns:
            Callable[P, T]: The wrapped function.
        """

        @functools.wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with self:
                return func(*args, **kwargs)

        return _wrapper

    def __enter__(self) -> None:
        """Save the current RNG states and reseed to the specified seed."""
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if self.use_cuda:
            self.cuda_state = torch.cuda.get_rng_state_all()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)  # includes cuda seeding

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        """Restore the saved RNG states before reseeding.

        Args:
            exc_type (Optional[Type[BaseException]]): The exception type should there be an exc.
            exc_value (Optional[BaseException]): The exception object should there be an exc.
            traceback (Optional[types.TracebackType]): The traceback should there be an exc.
        """
        random.setstate(self.random_state)  # type: ignore[misc]
        np.random.set_state(self.numpy_state)  # type: ignore[misc]
        torch.set_rng_state(self.torch_state)
        if self.use_cuda:
            torch.cuda.set_rng_state_all(self.cuda_state)


def set_determinism(check_hash_seed: bool = False) -> None:
    """Set all algorithms to be deterministic (may limit performance).

    Some PyTorch operations do not have a deterministic implementation. A warning will be raised \
    when such an operation is invoked.

    The PYTHONHASHSEED must be set before the Python process starts, so it's not set but only \
    checked here. It's an optional setting that guarantees hashing is not randomized.

    Args:
        check_hash_seed (bool, optional): Whether to check PYTHONHASHSEED. Defaults to False.
    """
    if check_hash_seed:
        assert os.environ.get("PYTHONHASHSEED", "") == "0", "PYTHONHASHSEED not set."
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
