from typing import Type, Union, cast

from . import layers as layers
from . import model as model
from . import region_graph as region_graph
from . import reparams as reparams
from . import symbolic as symbolic
from . import utils as utils


def set_layer_comp_space(comp_space: Union[Type[utils.ComputationSapce], str]) -> None:
    """Set the global computational space for layers.

    Args:
        comp_space (Union[Type[utils.ComputationSapce], str]): The computational space to use, can \
            be speficied by either the class or the name.
    """
    if isinstance(comp_space, str):
        comp_space = utils.ComputationSapce.get_comp_space_by_name(comp_space)

    # CAST: getattr gives Any.
    assert cast(
        bool, getattr(comp_space, "__final__", False)
    ), "A usable ComputationSapce must be final."

    layers.Layer.comp_space = comp_space


set_layer_comp_space(utils.LogSpace)
