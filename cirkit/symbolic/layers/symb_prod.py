from abc import ABC, abstractmethod
from typing import cast

from cirkit.symbolic.layers import SymbLayer


class SymbProdLayer(ABC, SymbLayer):
    """The abstract base class for symbolic product layers."""

    @staticmethod
    @abstractmethod
    def num_prod_units(num_units: int, arity: int) -> int:
        ...


class SymbHadamardLayer(SymbProdLayer):
    """The symbolic Hadamard product layer."""

    @staticmethod
    def num_prod_units(num_input_units: int, arity: int) -> int:
        return num_input_units


class SymbKroneckerLayer(SymbProdLayer):
    """The symbolic Kronecker product layer."""

    @staticmethod
    def num_prod_units(num_input_units: int, arity: int) -> int:
        return cast(int, num_input_units ** arity)
