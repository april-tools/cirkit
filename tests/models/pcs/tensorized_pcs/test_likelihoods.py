import itertools
from typing import Callable, Dict, List, Optional, Union

import pytest
import torch

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import UncollapsedCPLayer
from cirkit.models import TensorizedPC
from cirkit.models.functional import integrate
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils import RandomCtx
from cirkit.utils.type_aliases import ReparamFactory
from tests import floats


def get_deep_pc(  # type: ignore[misc]
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    reparam_func: ReparamFactory = ReparamIdentity,
) -> TensorizedPC:
    # TODO: type of kwargs should be refined
    rg = rg_cls(**kwargs)
    pc = TensorizedPC.from_region_graph(
        rg,
        layer_cls=UncollapsedCPLayer,
        efamily_cls=CategoricalLayer,
        layer_kwargs={"rank": 1},  # type: ignore[misc]
        efamily_kwargs={"num_categories": 2},  # type: ignore[misc]
        num_inner_units=16,
        num_input_units=16,
        reparam=reparam_func,
    )
    return pc


@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs,true_log_z",
    [
        (PoonDomingos, {"shape": [4, 4], "delta": 2}, 10.945370590340003),
        (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}, 51.23612793354572),
        (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}, 24.35743565443566),
        (PoonDomingos, {"shape": [3, 3], "delta": 2}, None),
        (QuadTree, {"width": 3, "height": 3, "struct_decomp": False}, None),
        (QuadTree, {"width": 3, "height": 3, "struct_decomp": True}, None),
        (RandomBinaryTree, {"num_vars": 9, "depth": 3, "num_repetitions": 2}, None),
    ],
)
@RandomCtx(42)
def test_pc_likelihoods(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
    true_log_z: Optional[float],
) -> None:
    """Tests the creation and EVI and MAR probabilistic inference on a PC.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
        true_log_z (Optional[float]): The answer of partition func. \
            NOTE: we don't know if it's correct, but it guarantees reproducibility.
    """
    pc = get_deep_pc(rg_cls, kwargs)  # type: ignore[misc]
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
    lls = log_scores - log_z

    # Check the partition function computation
    assert floats.isclose(log_z, torch.logsumexp(log_scores, dim=0, keepdim=True))

    # Compare the partition function against the answer, if given
    if true_log_z is not None:
        assert floats.isclose(log_z, torch.tensor(true_log_z)), f"{log_z.item()}"

    # Perform variable marginalization on the last two variables
    mar_data = all_data[::4]
    mar_scores = pc.integrate(mar_data, [num_vars - 2, num_vars - 1])
    mar_lls = mar_scores - log_z

    # Check the results of marginalization
    sum_lls = torch.logsumexp(lls.view(-1, 4), dim=1, keepdim=True)
    assert mar_lls.shape[0] == lls.shape[0] // 4 and len(mar_lls.shape) == len(lls.shape)
    assert floats.allclose(sum_lls, mar_lls)


# TODO: is fold_mask with multi-batch properly covered?
@pytest.mark.parametrize(  # type: ignore[misc]
    "rg_cls,kwargs",
    [
        (PoonDomingos, {"shape": [4, 4], "delta": 2}),
        (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}),
        (QuadTree, {"width": 4, "height": 4, "struct_decomp": True}),
        (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}),
    ],
)
@RandomCtx(42)
def test_pc_multi_batch(
    rg_cls: Callable[..., RegionGraph],
    kwargs: Dict[str, Union[int, bool, List[int]]],
) -> None:
    """Tests the creation and EVI probabilistic inference on a PC with mutiple batch dims.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
    """
    pc = get_deep_pc(rg_cls, kwargs)  # type: ignore[misc]
    num_vars = pc.num_vars

    # Generate all possible combinations of 16 integers from the list of possible values
    possible_values = [0, 1]
    all_data = torch.tensor(
        list(itertools.product(possible_values, repeat=num_vars))  # type: ignore[misc]
    ).view(32, 2048, 16, 1)

    # Instantiate the integral of the PC, i.e., computing the partition function
    pc_pf = integrate(pc)
    log_z = pc_pf(all_data)
    assert log_z.shape == (1, 1)  # TODO: is this shape correct? but is currently implemented

    # Compute outputs
    log_scores = pc(all_data)
    assert log_scores.shape == (32, 2048, 1)

    # TODO: mypy don't understand batch (iter of Tensor)
    batched_log_scores = torch.stack([pc(batch) for batch in all_data], dim=0)  # type: ignore[misc]
    assert torch.all(log_scores == batched_log_scores)
