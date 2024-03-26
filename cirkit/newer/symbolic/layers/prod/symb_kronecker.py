from cirkit.newer.symbolic.layers.prod.symb_prod import SymbProdLayer


class SymbKroneckerLayer(SymbProdLayer):
    """The symbolic Kronecker product layer."""

    @staticmethod
    def num_prod_units(num_units: int, arity: int) -> int:
        return num_units**arity
