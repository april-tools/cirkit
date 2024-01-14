import itertools
import math

import pytest

from cirkit.new.region_graph.algorithms.random_binary_tree import RandomBinaryTree
from cirkit.utils import RandomCtx


@pytest.mark.parametrize(
    "num_vars,depth,num_repetitions", list(itertools.product([8, 16], [0, 1, 3], [1, 3]))
)
def test_rg_random_binary_tree(num_vars: int, depth: int, num_repetitions: int) -> None:
    # Not enough variables for splitting at a certain depth
    # TODO: why set 42 without checking specific value?
    if num_vars < math.pow(2, depth):
        with pytest.raises(AssertionError), RandomCtx(42):
            RandomBinaryTree(num_vars=num_vars, depth=depth, num_repetitions= num_repetitions)
        return

    with RandomCtx(42):
        rg = RandomBinaryTree(num_vars=num_vars, depth=depth, num_repetitions= num_repetitions)
    print(type(num_vars))
    print(depth)
    print(num_repetitions)
    assert rg.is_smooth
    assert rg.is_decomposable
