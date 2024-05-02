from collections import defaultdict
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Dict, Iterable, Optional, Type

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import AbstractLayerOperator, Layer
from cirkit.symbolic.operators import (
    DEFAULT_OPERATOR_RULES,
    LayerOperatorFunc,
    LayerOperatorSign,
    LayerOperatorSpecs,
)


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

    def __str__(self) -> str:
        signature_repr = ", ".join(map(lambda cls: cls.__name__, self._signature))
        return f"Symbolic operator for signature ({signature_repr}) not found"


class OperatorRegistry(AbstractContextManager):
    def __init__(self):
        # The symbolic operator rule specifications, for each symbolic operator over layers
        self._rules: Dict[AbstractLayerOperator, LayerOperatorSpecs] = defaultdict(dict)

        # The token used to restore the operator registry context
        self._token: Optional[Token[OperatorRegistry]] = None

    @classmethod
    def from_default_rules(cls) -> "OperatorRegistry":
        registry = cls()
        for op, funcs in DEFAULT_OPERATOR_RULES.items():
            for f in funcs:
                registry.add_rule(op, f)
        return registry

    @property
    def operators(self) -> Iterable[AbstractLayerOperator]:
        return self._rules.keys()

    def __enter__(self) -> "OperatorRegistry":
        self._token = OPERATOR_REGISTRY.set(self)
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        OPERATOR_REGISTRY.reset(self._token)
        self._token = None
        return None

    def has_rule(self, op: AbstractLayerOperator, *signature: Type[Layer]) -> bool:
        if op not in self._rules:
            return False
        op_rules = self._rules[op]
        known_signatures = op_rules.keys()
        if signature in known_signatures:
            return True
        for s in known_signatures:
            if len(signature) != len(s):
                continue
            if all(issubclass(x[0], x[1]) for x in zip(signature, s)):
                return True
        return False

    def retrieve_rule(
        self, op: AbstractLayerOperator, *signature: Type[Layer]
    ) -> LayerOperatorFunc:
        if op not in self._rules:
            raise OperatorNotFound(op)
        op_rules = self._rules[op]
        known_signatures = op_rules.keys()
        if signature in known_signatures:
            return op_rules[signature]
        raise OperatorSignatureNotFound(*signature)

    def add_rule(self, op: AbstractLayerOperator, func: LayerOperatorFunc):
        args = func.__annotations__.copy()
        arg_names = args.keys()
        if "return" not in arg_names or not issubclass(args["return"], CircuitBlock):
            raise ValueError(
                f"The function is not an operator over symbolic layers.\nIdentifier: {func.__name__}\nAnnotations: {args}"
            )
        del args["return"]
        arg_names = list(args.keys())
        arg_types = [args[a] for a in arg_names]
        arg_layer_types = list(
            filter(
                lambda x: isinstance(x[1], type) and issubclass(x[1], Layer), enumerate(arg_types)
            )
        )
        arg_layer_types_locs, signature = zip(*arg_layer_types)
        if arg_layer_types_locs != tuple(range(len(arg_layer_types_locs))):
            raise ValueError(
                "The layer operands should be the first arguments of the operator rule function"
            )
        self._rules[op][signature] = func


# Context variable holding the current global operator registry.
# This is updated when entering an operator registry context.
OPERATOR_REGISTRY: ContextVar[OperatorRegistry] = ContextVar(
    "OPERATOR_REGISTRY", default=OperatorRegistry.from_default_rules()
)
