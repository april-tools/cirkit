from typing import Dict, FrozenSet, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

HyperCube = Tuple[Tuple[int, ...], Tuple[int, ...]]  # Just to shorten the annotation.
"""A hypercube represented by "top-left" and "bottom-right" coordinates (cut points)."""


class HypercubeToScope(Dict[HyperCube, FrozenSet[int]]):
    """Helper class to map sub-hypercubes to scopes with caching for variables arranged in a \
    hypercube.

    This is implemented as a dict subclass with customized __missing__, so that:
    - If a hypercube is already queried, the corresponding scope is retrieved the dict;
    - If it's not in the dict yet, the scope is calculated and cached to the dict.
    """

    def __init__(self, shape: Sequence[int]) -> None:
        """Init class.

        Note that this does not accept initial elements and is initialized empty.

        Args:
            shape (Sequence[int]): The shape of the whole hypercube.
        """
        self.ndims = len(shape)
        self.shape = tuple(shape)
        # We assume it's feasible to save the whole hypercube, since it should be the whole region.
        self.hypercube: NDArray[np.int64] = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)

    def __missing__(self, key: HyperCube) -> FrozenSet[int]:
        """Construct the item when not exist in the dict.

        Args:
            key (HyperCube): The key that is missing from the dict, i.e., a hypercube that is \
                visited for the first time.

        Returns:
            FrozenSet[int]: The value for the key, i.e., the corresponding scope.
        """
        point1, point2 = key  # HyperCube is from point1 to point2.

        assert (
            len(point1) == len(point2) == self.ndims
        ), "The shape of the HyperCube is not correct."
        assert all(
            0 <= x1 < x2 <= shape for x1, x2, shape in zip(point1, point2, self.shape)
        ), "The HyperCube is empty."

        # Ignore: Numpy has typing issues.
        return frozenset(
            self.hypercube[
                tuple(slice(x1, x2) for x1, x2 in zip(point1, point2))  # type: ignore[misc]
            ]
            .reshape(-1)
            .tolist()
        )
