# pylint: disable=too-many-locals
# TODO: add to pyproject.toml
import math
from typing import Dict, Literal, Tuple, Type

import torch
from torch import Tensor

from cirkit.new.layers import CategoricalLayer, CPLayer, InputLayer, NormalLayer
from cirkit.new.model import TensorizedCircuit
from cirkit.new.region_graph import RegionGraph, RegionNode
from cirkit.new.reparams import EFNormalReparam, ExpReparam, LogSoftmaxReparam, Reparameterization
from cirkit.new.symbolic import SymbolicTensorizedCircuit
from cirkit.new.utils.type_aliases import SymbCfgFactory


def get_two_circuits(
    *, same_scope: bool = True, setting: Literal["cat", "norm"] = "cat"
) -> Tuple[TensorizedCircuit, TensorizedCircuit]:
    # TODO: duplicated code?
    # Build RG
    rg1 = RegionGraph()
    node0 = RegionNode({0})
    node1 = RegionNode({1})
    node2 = RegionNode({2})
    node3 = RegionNode({3})
    region01 = RegionNode({0, 1})
    region23 = RegionNode({2, 3})
    region0123 = RegionNode({0, 1, 2, 3})
    rg1.add_partitioning(region01, (node0, node1))
    rg1.add_partitioning(region23, (node2, node3))
    rg1.add_partitioning(region0123, (region01, region23))
    rg1 = rg1.freeze()

    if same_scope:
        rg2 = rg1
    else:
        rg2 = RegionGraph()
        node0 = RegionNode({0})
        node1 = RegionNode({1})
        node4 = RegionNode({4})
        node5 = RegionNode({5})
        region01 = RegionNode({0, 1})
        region45 = RegionNode({4, 5})
        region0145 = RegionNode({0, 1, 4, 5})
        rg2.add_partitioning(region01, (node0, node1))
        rg2.add_partitioning(region45, (node4, node5))
        rg2.add_partitioning(region0145, (region01, region45))
        rg2 = rg2.freeze()

    # Build symbolic circuit
    num_channels = 1
    num_classes = 1
    inner_cls = CPLayer
    inner_kwargs: Dict[str, None] = {}
    inner_reparam = ExpReparam

    num_units_1 = 4
    num_units_2 = 5

    input_cls: Type[InputLayer]
    input_reparam: Type[Reparameterization]

    if setting == "cat":
        input_cls = CategoricalLayer
        input_kwargs = {"num_categories": 2}
        input_reparam = LogSoftmaxReparam
    elif setting == "norm":
        input_cls = NormalLayer
        input_kwargs = {}
        input_reparam = EFNormalReparam
    else:
        assert False, "This should not happen."

    symbolic_circuit_1 = SymbolicTensorizedCircuit(
        rg1,
        num_input_units=num_units_1,
        num_sum_units=num_units_1,
        num_channels=num_channels,
        num_classes=num_classes,
        input_cfg=SymbCfgFactory(
            layer_cls=input_cls, layer_kwargs=input_kwargs, reparam_factory=input_reparam
        ),
        sum_cfg=SymbCfgFactory(
            layer_cls=inner_cls, layer_kwargs=inner_kwargs, reparam_factory=inner_reparam
        ),
        prod_cfg=SymbCfgFactory(layer_cls=inner_cls, layer_kwargs=inner_kwargs),
    )

    symbolic_circuit_2 = SymbolicTensorizedCircuit(
        rg2,
        num_input_units=num_units_2,
        num_sum_units=num_units_2,
        num_channels=num_channels,
        num_classes=num_classes,
        input_cfg=SymbCfgFactory(
            layer_cls=input_cls, layer_kwargs=input_kwargs, reparam_factory=input_reparam
        ),
        sum_cfg=SymbCfgFactory(
            layer_cls=inner_cls, layer_kwargs=inner_kwargs, reparam_factory=inner_reparam
        ),
        prod_cfg=SymbCfgFactory(layer_cls=inner_cls, layer_kwargs=inner_kwargs),
    )

    return (TensorizedCircuit(symbolic_circuit_1), TensorizedCircuit(symbolic_circuit_2))


def pf_of_product_of_normal(eta: Tensor) -> Tensor:
    """This function calculates the partition function of the product of two normal distributions, \
    to verify if the implementation in cirkit is correct.

    By definition:
        LogInt N(x; η0,η1)*N(x; η2,η3) dx \
        = ((η0*η3-η1*η2)^2 / 4*η1*η3*(η1+η3)) - 0.5*log(pi) - 0.5*log((-η1-η3)/(η1*η3)).
        
    Args:
        eta (Tensor): The parameters of two Gaussians, shape (H, K*K, 4).

    Returns:
        Tensor: The partition functions, shape (K*K,).
    """
    log_sq_pi = 0.5 * math.log(math.pi)
    square_term = (
        eta[:, :, 0] * eta[:, :, 3] - eta[:, :, 1] * eta[:, :, 2]  # type: ignore[misc]
    ) ** 2
    exponent = square_term / (  # type: ignore[misc]
        4 * eta[:, :, 1] * eta[:, :, 3] * (eta[:, :, 1] + eta[:, :, 3])
    )
    normalizer = 0.5 * torch.log((-eta[:, :, 1] - eta[:, :, 3]) / (eta[:, :, 1] * eta[:, :, 3]))
    return torch.sum(exponent - log_sq_pi - normalizer, dim=0)  # type: ignore[misc]
