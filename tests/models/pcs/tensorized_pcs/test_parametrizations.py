import functools
import itertools
from typing import Callable, Dict, List, Union

import pytest
import torch

from cirkit.models.functional import integrate
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree
from cirkit.reparams.leaf import ReparamClamp, ReparamExp, ReparamSoftmax, ReparamSquare
from cirkit.utils import RandomCtx
from cirkit.utils.type_aliases import ReparamFactory
from tests import floats
from tests.models.pcs.tensorized_pcs.test_likelihoods import get_deep_pc


@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs,reparam_name",
    [
        (rg_cls, kwargs, reparam_name)
        for (rg_cls, kwargs) in (
            (PoonDomingos, {"shape": [4, 4], "delta": 2}),
            (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}),
            (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}),
            (PoonDomingos, {"shape": [3, 3], "delta": 2}),
            (QuadTree, {"width": 3, "height": 3, "struct_decomp": False}),
            (QuadTree, {"width": 3, "height": 3, "struct_decomp": True}),
            (RandomBinaryTree, {"num_vars": 9, "depth": 3, "num_repetitions": 2}),
        )
        for reparam_name in ("exp", "square", "softmax", "positive")
    ],
)
@RandomCtx(42)
def test_pc_nonneg_reparams(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    reparam_name: str,
) -> None:
    """Tests multiple non-negative re-parametrizations on tensorized circuits.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
        reparam_name (str): The reparametrization function identifier.
    """
    reparam: ReparamFactory
    if reparam_name == "exp":
        reparam = ReparamExp
    elif reparam_name == "square":
        reparam = ReparamSquare
    elif reparam_name == "softmax":
        reparam = ReparamSoftmax
    elif reparam_name == "positive":
        reparam = functools.partial(ReparamClamp, min=1e-7)  # type: ignore[misc]
    else:
        assert False

    pc = get_deep_pc(rg_cls, kwargs, reparam_func=reparam)  # type: ignore[misc]
    num_vars = pc.num_vars

    # Generate all possible combinations of 16 integers from the list of possible values
    possible_values = [0, 1]
    all_data = torch.tensor(
        list(itertools.product(possible_values, repeat=num_vars))  # type: ignore[misc]
    ).unsqueeze(dim=-1)

    # Instantiate the integral of the PC, i.e., computing the partition function
    pc_pf = integrate(pc)
    log_z = pc_pf(all_data)
    assert log_z.shape == (1, 1)

    # Compute outputs
    log_scores = pc(all_data)

    # Check the partition function computation
    assert floats.isclose(log_z, torch.logsumexp(log_scores, dim=0, keepdim=True))

    # The circuit should be already normalized,
    #  if the re-parameterization is via softmax and using normalized input distributions
    if reparam_name == "softmax":
        assert floats.allclose(log_z, 0.0)
