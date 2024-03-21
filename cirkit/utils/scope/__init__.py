from .fronzenset_scope import FrozenSetScope as FrozenSetScope
from .scope import Scope as Scope

Scope.impl = FrozenSetScope
