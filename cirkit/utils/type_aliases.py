from typing import Dict, List, TypedDict, Union
from typing_extensions import TypeAlias  # FUTURE: in typing from 3.10, deprecated in 3.12

# Here're the type defs and aliases shared across the library. The code should work without anything
# from this file if we don't care about typing. That is, we don't define runtime behaviour here.
# If a type is private and only used in one file, it can also be defined in that file.


# The allowed value types are what can be saved in json.
RGNodeMetadata: TypeAlias = Dict[str, Union[int, float, str, bool]]
"""The type for the metadata of RGNode."""


class RegionDict(TypedDict):
    """The structure of a region node in the json file."""

    scope: List[int]  # The scope of this region node, specified by id of variable.


class PartitionDict(TypedDict):
    """The structure of a partition node in the json file."""

    inputs: List[int]  # The inputs of this partition node, specified by id of region node.
    output: int  # The output of this partition node, specified by id of region node.


class RegionGraphJson(TypedDict):
    """The structure of the region graph json file."""

    # The regions of RG represented by a mapping from id in str to either a dict or only the scope.
    regions: Dict[str, Union[RegionDict, List[int]]]
    # The graph of RG represented by a list of partitions.
    graph: List[PartitionDict]
