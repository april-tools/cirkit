from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, Type, TypeVar

RegistrySign = TypeVar("RegistrySign")
RegistryFunc = TypeVar("RegistryFunc")


class InvalidRule(Exception):
    def __init__(self, annotations: Dict[str, Type]):
        super().__init__(f"Compilation rule with annotations '{annotations} is invalid")


class CompilationRuleNotFound(Exception):
    def __init__(self, signature: RegistrySign):
        super().__init__(f"Compilation rule for signature '{signature}' not found")


class CompilerRegistry(Generic[RegistrySign, RegistryFunc], ABC):
    def __init__(self, rules: Optional[Dict[RegistrySign, RegistryFunc]] = None):
        self._rules = {} if rules is None else rules

    @classmethod
    @abstractmethod
    def _validate_rule_signature(cls, func: RegistryFunc) -> Optional[RegistrySign]:
        ...

    def add_rule(self, func: RegistryFunc) -> None:
        sigature = self._validate_rule_signature(func)
        if sigature is None:
            raise InvalidRule(func.__annotations__)
        self._rules[sigature] = func

    def retrieve_rule(self, signature: RegistrySign) -> RegistryFunc:
        func = self._rules.get(signature, None)
        if func is not None:
            return func
        raise CompilationRuleNotFound(signature)
