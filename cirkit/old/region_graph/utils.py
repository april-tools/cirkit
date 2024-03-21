from typing import Dict, Sequence, Set, Tuple

import numpy as np
from numpy._typing import NDArray

HyperCube = Tuple[Tuple[int, ...], Tuple[int, ...]]


class HypercubeScopeCache:  # pylint: disable=too-few-public-methods
    """Helper class for building region graph structures. Represents a function cache, \
        mapping hypercubes to their unrolled scope.

    For example consider the hypercube ((0, 0), (4, 5)), which is a rectangle \
        with 4 rows and 5 columns. We assign \
        linear indices to the elements in this rectangle as follows:
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
    Similarly, we assign linear indices to higher-dimensional hypercubes, \
        where higher axes toggle faster than lower \
        axes. The scope of sub-hypercubes are just the unrolled linear indices. \
        For example, for the rectangle above, \
        the sub-rectangle ((1, 2), (4, 5)) has scope (7, 8, 9, 12, 13, 14, 17, 18, 19).

    This class just represents a cached mapping from hypercubes to their scopes.
    """

    def __init__(self) -> None:
        """Init class."""
        self._hyper_cube_to_scope: Dict[HyperCube, Set[int]] = {}

    def __call__(self, hypercube: HyperCube, shape: Sequence[int]) -> Set[int]:
        """Get the scope of a hypercube.

        Args:
            hypercube (Tuple[Tuple[int,...],Tuple[int,...]]): The hypercube.
            shape (Sequence[int]): The total shape.

        Returns:
            Set[int]: Corresponding scope.
        """
        # TODO: accept tuple of seq? but must be hashable
        # TODO: return must be hashable. rewrite?
        if hypercube in self._hyper_cube_to_scope:
            return self._hyper_cube_to_scope[hypercube]

        x1 = hypercube[0]
        x2 = hypercube[1]

        assert len(x1) == len(x2) and len(x1) == len(shape)
        # TODO: should rewrite, also the following in tuple comp
        assert all(x1[i] >= 0 and x2[i] <= shape[i] for i in range(len(shape)))

        scope: NDArray[np.int64] = np.zeros(
            tuple(x2[i] - x1[i] for i in range(len(shape))), dtype=np.int64
        )
        f = 1
        for i, c in enumerate(reversed(range(len(shape)))):
            range_to_add: NDArray[np.int64] = f * np.array(range(x1[c], x2[c]), np.int64)
            # TODO: find a better way to reshape
            scope += np.reshape(range_to_add, (len(range_to_add),) + i * (1,))
            f *= shape[c]

        scope_: Set[int] = set(scope.reshape(-1).tolist())  # type: ignore[misc]
        self._hyper_cube_to_scope[hypercube] = scope_
        return scope_
