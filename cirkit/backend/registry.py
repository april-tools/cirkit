from abc import ABC, abstractmethod
from typing import Generic, TypeVar

RegistrySign = TypeVar("RegistrySign")
RegistryFunc = TypeVar("RegistryFunc")


class InvalidRuleSign(Exception):
    def __init__(self, annotations: dict[str, type]):
        super().__init__(
            f"Cannot extract rule signature from function with annotations '{annotations}"
        )


class InvalidRuleFunction(Exception):
    def __init__(self, annotations: dict[str, type]):
        super().__init__(f"Invalid Compilation rule function with annotations '{annotations}'")


class CompilationRuleNotFound(Exception):
    def __init__(self, signature: RegistrySign):
        super().__init__(f"Compilation rule for signature '{signature}' not found")


class CompilerRegistry(Generic[RegistrySign, RegistryFunc], ABC):
    def __init__(self, rules: dict[RegistrySign, RegistryFunc] | None = None):
        self._rules = {} if rules is None else rules

    @classmethod
    def _validate_rule_function(cls, func: RegistryFunc) -> bool:
        return True

    @classmethod
    def _retrieve_signature(cls, func: RegistryFunc) -> RegistrySign:
        raise InvalidRuleSign(func.__annotations__)

    @property
    def signatures(self) -> list[RegistrySign]:
        return list(self._rules)

    def add_rule(self, func: RegistryFunc, *, signature: RegistrySign | None = None) -> None:
        if not self._validate_rule_function(func):
            raise InvalidRuleFunction(func.__annotations__)
        if signature is None:
            signature = self._retrieve_signature(func)
        self._rules[signature] = func

    def retrieve_rule(self, signature: RegistrySign) -> RegistryFunc:
        func = self._rules.get(signature, None)
        if func is not None:
            return func
        raise CompilationRuleNotFound(signature)
