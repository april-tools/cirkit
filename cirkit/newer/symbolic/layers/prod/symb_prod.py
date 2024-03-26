from abc import ABC, abstractmethod

from cirkit.newer.symbolic.layers.symb_layer import SymbLayer


class SymbProdLayer(ABC, SymbLayer):
    """The abstract base class for symbolic product layers."""

    @staticmethod
    @abstractmethod
    def num_prod_units(num_units: int, arity: int) -> int:
        ...
