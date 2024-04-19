from collections import defaultdict
from typing import Callable, Dict, Iterable, Optional, Tuple, Type

from cirkit.symbolic.layers import AbstractLayerOperator, ExpFamilyLayer, Layer, LayerOperation
from cirkit.symbolic.operators import integrate_ef_layer

LayerOperatorSignature = Tuple[Type[Layer], ...]
LayerOperatorFunction = Callable[..., Layer]  # TODO: add typed ellipsis (>=py3.10)
LayerOperatorSpecs = Dict[LayerOperatorSignature, LayerOperatorFunction]


_DEFAULT_OPERATOR_RULES: Dict[AbstractLayerOperator, LayerOperatorSpecs] = {
    LayerOperation.INTEGRATION: {(ExpFamilyLayer,): integrate_ef_layer},  # TODO: fill
    LayerOperation.DIFFERENTIATION: {},  # TODO: fill
    LayerOperation.MULTIPLICATION: {},  # TODO: fill
}


class OperatorRegistry:
    def __init__(self):
        self._rules: Dict[AbstractLayerOperator, LayerOperatorSpecs] = defaultdict(dict)
        self._rules.update(_DEFAULT_OPERATOR_RULES)

    @property
    def operators(self) -> Iterable[AbstractLayerOperator]:
        return self._rules.keys()

    def has_rule(self, op: AbstractLayerOperator, *symb_cls: Type[Layer]) -> bool:
        if op not in self._rules:
            return False
        op_rules = self._rules[op]
        signature = tuple(symb_cls)
        if signature not in op_rules:
            return False
        return True

    def retrieve_rule(
        self, op: AbstractLayerOperator, *symb_cls: Type[Layer]
    ) -> LayerOperatorFunction:
        if op not in self._rules:
            raise IndexError(f"The operator '{op}' is unknown")
        op_rules = self._rules[op]
        signature = tuple(symb_cls)
        if signature not in op_rules:
            raise IndexError(f"An operator for the signature '{signature}' has not been found")
        return op_rules[signature]

    def register_rule(
        self,
        op: AbstractLayerOperator,
        func: LayerOperatorFunction,
        commutative: Optional[bool] = None,
    ):
        args = func.__annotations__
        arg_names = list(filter(lambda a: a != "return", args.keys()))
        if len(arg_names) == 0 or not all(issubclass(args[a], Layer) for a in arg_names):
            raise ValueError("The function is not an operator over symbolic layers")
        if (
            len(arg_names) == 2
        ):  # binary operator (special case as to deal with commutative operators)
            lhs_cls = args[arg_names[0]]
            rhs_cls = args[arg_names[1]]
            self._rules[op][(lhs_cls, rhs_cls)] = func
            if commutative and lhs_cls != rhs_cls:
                self._rules[op][(rhs_cls, lhs_cls)] = lambda rhs, lhs: func(lhs, rhs)
        else:  # n-ary operator
            signature = tuple(args[a] for a in arg_names)
            self._rules[op][signature] = func
