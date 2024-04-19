import itertools
import math

import pytest

from cirkit.templates.region_graph import RandomBinaryTree
from tests.templates.region_graph.test_region_graph import (
    check_region_graph_save_load
)


@pytest.mark.parametrize(
    "num_vars,depth,num_repetitions", list(itertools.product([1, 8, 16], [0, 1, 3], [1, 3]))
)
def test_rg_random_binary_tree(num_vars: int, depth: int, num_repetitions: int) -> None:
    # Not enough variables for splitting at a certain depth
    # TODO: why set 42 without checking specific value?
    if num_vars < math.pow(2, depth):
        with pytest.raises(AssertionError):
            RandomBinaryTree(num_vars, depth, num_repetitions)
        return

    rg = RandomBinaryTree(num_vars, depth, num_repetitions)
    assert rg.is_smooth
    assert rg.is_decomposable
    check_region_graph_save_load(rg)
