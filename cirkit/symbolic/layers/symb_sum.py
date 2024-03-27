from abc import ABC

from cirkit.symbolic.layers.symb_layer import SymbLayer


class SymbSumLayer(ABC, SymbLayer):
    """The abstract base class for symbolic sum layers."""


class SymbDenseLayer(SymbSumLayer):
    """The symbolic dense sum layer."""


class SymbMixingLayer(SymbSumLayer):
    """The symbolic mixing sum layer."""
