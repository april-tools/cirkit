from cirkit.symbolic.layers import Layer

GateFunctionSpecs = dict[str, list[Layer]]
"""The gate function specification. It is a map from an id (a string) to the list of layers 
it should parametrize."""


GateFunctionParameterSpecs = dict[str, tuple[int, ...]]
"""The gate function parameter specification.
It is a map from a gate function name to a parameter group specification.
The parameter group specification is the shapes that must be computed for that parameter.
Groups are constructed by collating together parameters with the same shape."""