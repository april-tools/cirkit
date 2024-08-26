import itertools

import pytest

from cirkit.templates.region_graph import FullyFactorized, RegionNode
from cirkit.utils.scope import Scope
from tests.templates.region_graph.test_utils import check_region_graph_save_load


@pytest.mark.parametrize(
    "num_variables,num_repetitions",
    itertools.product([1, 5], [1, 3]),
)
def test_rg_algorithm_fully_factorized(num_variables: int, num_repetitions: int):
    rg = FullyFactorized(num_variables, num_repetitions=num_repetitions)
    assert rg.is_omni_compatible
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert all(len(rg.partition_inputs(ptn)) == num_variables for ptn in rg.partition_nodes)
    if num_variables > 1:
        assert len(rg.region_inputs(root)) == num_repetitions
    check_region_graph_save_load(rg)
