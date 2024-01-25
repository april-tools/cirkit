# mypy: disable-error-code="assignment"
# IGNORE: For this file we ignore the above because we incompatibly force-assign the methods.

from typing import FrozenSet
from typing_extensions import final  # FUTURE: in typing from 3.11 for __final__

from cirkit.new.utils.scope.scope import Scope


# IGNORE: The annotation for final in typeshed/typing_extensions.pyi contains Any.
# IGNORE: Incompatible multi-inheritance is expected.
@Scope.register("frozenset")  # type: ignore[misc]
@final  # type: ignore[misc]
class FrozenSetScope(Scope, FrozenSet[int]):  # type: ignore[misc]
    """The Scope implemented by frozenset."""

    __new__ = frozenset.__new__
    __contains__ = frozenset.__contains__
    __iter__ = frozenset.__iter__
    __len__ = frozenset.__len__
    __le__ = frozenset.__le__
    __ge__ = frozenset.__ge__
    __and__ = frozenset.__and__
    __or__ = frozenset.__or__
