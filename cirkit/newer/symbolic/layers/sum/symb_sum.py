from abc import ABC

from cirkit.newer.symbolic.layers.symb_layer import SymbLayer


class SymbSumLayer(ABC, SymbLayer):
    """The abstract base class for symbolic sum layers."""
