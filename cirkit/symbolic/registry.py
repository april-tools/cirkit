from collections import defaultdict
from typing import Type, Tuple, Callable, Iterable, Optional, Dict

from cirkit.symbolic.operators import integrate_input_layer, integrate_ef_layer
from cirkit.symbolic.symb_layers import SymbLayer, AbstractSymbLayerOperator, SymbLayerOperator, SymbExpFamilyLayer, \
    SymbInputLayer

SymbLayerOperatorSignature = Tuple[Type[SymbLayer], ...]
SymbLayerOperatorFunction = Callable[..., SymbLayer]  # TODO: add typed ellipsis (>=py3.10)
SymbLayerOperatorSpecs = Dict[SymbLayerOperatorSignature, SymbLayerOperatorFunction]


_DEFAULT_SYMB_OPERATOR_RULES: Dict[AbstractSymbLayerOperator, SymbLayerOperatorSpecs] = {
    SymbLayerOperator.INTEGRATION: {
        (SymbInputLayer,): integrate_input_layer,
        (SymbExpFamilyLayer,): integrate_ef_layer,
    },  # TODO: fill
    SymbLayerOperator.DIFFERENTIATION: {},  # TODO: fill
    SymbLayerOperator.KRONECKER: {}  # TODO: fill
}


class SymbOperatorRegistry:
    def __init__(self):
        self._rules: Dict[AbstractSymbLayerOperator, SymbLayerOperatorSpecs] = defaultdict(dict)
        self._rules.update(_DEFAULT_SYMB_OPERATOR_RULES)

    @property
    def operators(self) -> Iterable[AbstractSymbLayerOperator]:
        return self._rules.keys()

    def has_rule(self, op: AbstractSymbLayerOperator, *symb_cls: Type[SymbLayer]) -> bool:
        if op not in self._rules:
            return False
        op_rules = self._rules[op]
        signature = tuple(symb_cls)
        if signature not in op_rules:
            return False
        return True

    def retrieve_rule(self, op: AbstractSymbLayerOperator, *symb_cls: Type[SymbLayer]) -> SymbLayerOperatorFunction:
        if op not in self._rules:
            raise IndexError(f"The operator '{op}' is unknown")
        op_rules = self._rules[op]
        signature = tuple(symb_cls)
        if signature not in op_rules:
            raise IndexError(f"An operator for the signature '{signature}' has not been found")
        return op_rules[signature]

    def register_rule(
            self,
            op: AbstractSymbLayerOperator,
            func: SymbLayerOperatorFunction,
            commutative: Optional[bool] = None
    ):
        args = func.__annotations__
        arg_names = list(filter(lambda a: a != 'return', args.keys()))
        if len(arg_names) == 0 or not all(issubclass(args[a], SymbLayer) for a in arg_names):
            raise ValueError("The function is not an operator over symbolic layers")
        if len(arg_names) == 2:  # binary operator (special case as to deal with commutative operators)
            lhs_symb_cls = args[arg_names[0]]
            rhs_symb_cls = args[arg_names[1]]
            self._rules[op][(lhs_symb_cls, rhs_symb_cls)] = func
            if commutative and lhs_symb_cls != rhs_symb_cls:
                self._rules[op][(rhs_symb_cls, lhs_symb_cls)] = lambda rhs, lhs: func(lhs, rhs)
        else:  # n-ary operator
            symb_signature = tuple(args[a] for a in arg_names)
            self._rules[op][symb_signature] = func
