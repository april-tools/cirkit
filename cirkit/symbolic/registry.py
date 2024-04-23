from collections import defaultdict
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import AbstractLayerOperator, ExpFamilyLayer, Layer, LayerOperation
from cirkit.symbolic.operators import *

LayerOperatorSign = Tuple[Type[Layer], ...]
LayerOperatorFunc = Callable[..., CircuitBlock]
LayerOperatorSpecs = Dict[LayerOperatorSign, LayerOperatorFunc]


_DEFAULT_COMMUTATIVE_OPERATORS = [LayerOperation.MULTIPLICATION]


_DEFAULT_OPERATOR_RULES: Dict[AbstractLayerOperator, List[LayerOperatorFunc]] = {
    LayerOperation.INTEGRATION: [
        integrate_ef_layer,
    ],
    LayerOperation.DIFFERENTIATION: [],
    LayerOperation.MULTIPLICATION: [
        multiply_hadamard_layers,
        multiply_kronecker_layers,
        multiply_dense_layers,
        multiply_mixing_layers,
    ],
}


class OperatorNotFound(Exception):
    def __init__(self, op: AbstractLayerOperator):
        super().__init__()
        self._operator = op

    def __repr__(self) -> str:
        return f"Symbolic operator named '{self._operator.name}' not found"


class OperatorSignatureNotFound(Exception):
    def __init__(self, *signature: Type[Layer]):
        super().__init__()
        self._signature = tuple(signature)

    def __repr__(self) -> str:
        signature_repr = ", ".join(map(lambda cls: cls.__name__, self._signature))
        return f"Symbolic operator for signature ({signature_repr}) not found"


class OperatorRegistry:
    _DEFAULT_REGISTRY: Optional["OperatorRegistry"] = None

    def __init__(self):
        self._rules: Dict[AbstractLayerOperator, LayerOperatorSpecs] = {}

    @classmethod
    def from_default_rules(cls) -> "OperatorRegistry":
        if OperatorRegistry._DEFAULT_REGISTRY is None:
            registry = cls()
            for op, funcs in _DEFAULT_OPERATOR_RULES.items():
                for f in funcs:
                    registry.register_rule(op, f, commutative=op in _DEFAULT_COMMUTATIVE_OPERATORS)
            OperatorRegistry._DEFAULT_REGISTRY = registry
        return OperatorRegistry._DEFAULT_REGISTRY

    @property
    def operators(self) -> Iterable[AbstractLayerOperator]:
        return self._rules.keys()

    def has_rule(self, op: AbstractLayerOperator, *signature: Type[Layer]) -> bool:
        if op not in self._rules:
            return False
        op_rules = self._rules[op]
        signature = tuple(signature)
        if signature not in op_rules:
            return False
        return True

    def retrieve_rule(
        self, op: AbstractLayerOperator, *signature: Type[Layer]
    ) -> LayerOperatorFunc:
        if op not in self._rules:
            raise OperatorNotFound(op)
        op_rules = self._rules[op]
        signature = tuple(signature)
        if signature not in op_rules:
            raise OperatorSignatureNotFound(*signature)
        return op_rules[signature]

    def register_rule(
        self,
        op: AbstractLayerOperator,
        func: LayerOperatorFunc,
        commutative: Optional[bool] = None,
    ):
        args = func.__annotations__
        arg_names = list(args.keys())
        if (
            len(arg_names) == 0
            or "return" not in arg_names
            or not issubclass(args["return"], CircuitBlock)
        ):
            raise ValueError("The function is not an operator over symbolic layers")
        if (
            len(arg_names) == 3
            and issubclass(args[arg_names[0]], Layer)
            and issubclass(args[arg_names[1]], Layer)
        ):
            # Binary operator found (special case as to deal with commutative operators)
            lhs_cls = args[arg_names[0]]
            rhs_cls = args[arg_names[1]]
            self._rules[op][(lhs_cls, rhs_cls)] = func
            if commutative and lhs_cls != rhs_cls:
                self._rules[op][(rhs_cls, lhs_cls)] = lambda rhs, lhs: func(lhs, rhs)
        else:  # n-ary operator
            signature = tuple(args[a] for a in arg_names if issubclass(args[a], Layer))
            self._rules[op][signature] = func
