from typing import Iterable, List, Optional

from cirkit.symbolic.block import CircuitBlock
from cirkit.symbolic.layers import (
    ConstantLayer,
    ExpFamilyLayer,
    HadamardLayer,
    IndexLayer,
    KroneckerLayer,
)
from cirkit.utils.scope import Scope


def integrate_ef_layer(sl: ExpFamilyLayer, scope: Optional[Iterable[int]] = None) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    # Symbolically integrate an exponential family layer, which is a constant layer
    sl = ConstantLayer(sl.scope, sl.num_output_units, sl.num_channels, value=1.0)
    return CircuitBlock.from_layer(sl)


def multiply_hadamard_layers(lhs: HadamardLayer, rhs: HadamardLayer) -> CircuitBlock:
    sl = HadamardLayer(
        lhs.scope | rhs.scope,
        lhs.num_input_units * rhs.num_input_units,
        arity=max(lhs.arity, rhs.arity),
    )
    return CircuitBlock.from_layer(sl)


def multiply_kronecker_layers(lhs: KroneckerLayer, rhs: KroneckerLayer) -> CircuitBlock:
    sl = KroneckerLayer(
        lhs.scope | rhs.scope,
        lhs.num_input_units * rhs.num_input_units,
        arity=max(lhs.arity, rhs.arity),
    )
    # The product of kronecker layers is a kronecker layer followed by a permutation
    idx: List[int] = []  # TODO
    sil = IndexLayer(lhs.scope | rhs.scope, sl.num_output_units, sl.num_output_units, indices=idx)
    return CircuitBlock.from_layer_composition(sl, sil)
