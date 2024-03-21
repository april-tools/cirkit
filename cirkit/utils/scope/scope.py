from typing import (
    Callable,
    ClassVar,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    Type,
    TypeVar,
    cast,
    final,
)
from typing_extensions import Never  # FUTURE: in typing from 3.11

# NOTE: Explaination on the magic in class customization for Scope.
#       - It's an abstract class, because it does not include a complete implementation and is not
#           meant to be instantiated.
#       - It's not an abstract class, because instantiation is not prohibited, both in runtime and
#           in static checking.
#       - NotImplementedError is raised in "abstract" methods, so that this class works as an \
#           interface but is instantiable:
#           - All the methods are concrete and does not block instantiation;
#           - All the methods run into error when called so must be implemented by subclasses;
#       - The instantiation of the class is intercepted in __new__, so that:
#           - It behaves as if this class itself is instantiated;
#           - Different subclass implementations can be selected for the actual object so that those
#               not-implemented interface methods are never actually called.


ScopeClsT = TypeVar("ScopeClsT", bound=Type["Scope"])


class Scope(Collection[int], Hashable):
    """An immutable container (Hashable Collection) for a set of int to represent the scope of a \
    node in the region graph or a unit in the circuit.

    A scope should always be a subset of range(num_vars), but for efficiency this is not checked.
    """

    # NOTE: This is not set here by default, and a default value should be provided later.
    impl: ClassVar[Type["Scope"]]
    """The currently selected implementation."""

    _registry: ClassVar[Dict[str, Type["Scope"]]] = {}

    @final
    @staticmethod
    def register(name: str) -> Callable[[ScopeClsT], ScopeClsT]:
        """Register a concrete Scope implementation by its name.

        Args:
            name (str): The name to register.

        Returns:
            Callable[[ScopeClsT], ScopeClsT]: The class decorator to register a subclass.
        """

        def _decorator(cls: ScopeClsT) -> ScopeClsT:
            """Register a concrete Scope implementation by its name.

            Args:
                cls (ScopeClsT): The Scope subclass to register.

            Returns:
                ScopeClsT: The class passed in.
            """
            # CAST: getattr gives Any.
            assert cast(
                bool, getattr(cls, "__final__", False)
            ), "Subclasses of Scope should be final."

            methods_to_implement = ("__new__", "__contains__", "__iter__", "__len__")
            for method in methods_to_implement:
                # DISABLE: We are comparing callable objects.
                # IGNORE: getattr gives Any.
                # pylint: disable-next=comparison-with-callable
                assert getattr(cls, method) != getattr(  # type: ignore[misc]
                    Scope, method
                ), f"{cls} should implement {method}()."

            # Check final during runtime in case of multi-inheritance, which static checking cannot
            # capture and warn.
            methods_to_preserve = ("__init__", "__setattr__", "__delattr__", "__repr__")
            for method in methods_to_preserve:
                # IGNORE: getattr gives Any.
                assert getattr(cls, method) == getattr(  # type: ignore[misc]
                    Scope, method
                ), f"{cls} should not redefine final method {method}()."

            # __eq__ and __hash__ are not checked because they just follow the standard protocol.
            methods_to_override = (
                "__lt__",
                "__gt__",
                "__le__",
                "__ge__",
                "__and__",
                "__or__",
                "union",
            )
            for method in methods_to_override:
                # IGNORE: vars and getattr give Any.
                assert method in vars(cls) or getattr(cls, method) == getattr(  # type: ignore[misc]
                    Scope, method
                ), f"{cls} should either explicitly override {method}() or inherit from Scope."

            Scope._registry[name] = cls
            return cls

        return _decorator

    @final
    @staticmethod
    def list_all_scope_impl() -> Iterable[str]:
        """List all names of Scope implementations registered.

        Returns:
            Iterable[str]: An iterable over all names available.
        """
        return iter(Scope._registry)

    @final
    @staticmethod
    def set_scope_impl_by_name(name: str) -> None:
        """Set the active implementation to a Scope subclass specified by its registered name.

        Args:
            name (str): The name of implementation.
        """
        Scope.impl = Scope._registry[name]

    ################################################################################################
    # The following are about immutability, including constriction, disabling modification, etc.

    # We should use __new__ instead of __init__ because it's immutable.
    # NOTE: Subclasses only need to handle the initialization of internal data structure in __new__.
    def __new__(cls, scope: Iterable[int]) -> "Scope":
        """Construct a scope object.

        Args:
            scope (Iterable[int]): The scope as an iterable of variable ids. If already an object \
                of the same class, the object will be directly returned.

        Returns:
            Scope: A Scope object.
        """
        if isinstance(scope, Scope.impl):  # Reuse the same object when possible.
            return scope
        if cls == Scope.impl:  # If called from Scope.impl.__new__(), break infinite recursion.
            return super().__new__(cls)
        return Scope.impl.__new__(Scope.impl, scope)  # Enforce instantiation of the selected impl.

    # Mark __init__ as final so that subclasses cannot use it.
    # NOTE: This is just a no-op. The docstirng is only for intellisense.
    @final
    def __init__(self, scope: Iterable[int]) -> None:
        """Instantiate a Scope.

        Args:
            scope (Iterable[int]): The scope as an iterable of variable ids. If already an object \
                of the same class, the object will be directly returned.
        """
        # Strictly no-op, even no super().__init__(), which may be wrong in multi-inheritance.

    # setattr and delattr are disabled for immutablility.
    # NOTE: It's possible to bypass through object.__setattr__ and object.__delattr__, e.g. for use
    #       in the __new__ of subclasses. But users should not use like that.
    @final
    def __setattr__(self, name: str, value: object) -> Never:
        """Intercept the attribute assignment.

        Raises:
            TypeError: When an attribute is assigned to Scope.
        """
        raise TypeError("Scope is expected to be immutable.")

    @final
    def __delattr__(self, name: str) -> Never:
        """Intercept the attribute deletion.

        Raises:
            TypeError: When an attribute is deleted to Scope.
        """
        raise TypeError("Scope is expected to be immutable.")

    # We hide the actual subclass and just show everything as Scope.
    @final
    def __repr__(self) -> str:
        """Generate the repr string of the scope, for repr().

        Returns:
            str: The str representation of the scope.
        """
        return f"Scope({repr(set(self))})"  # Scope({0, 1, ...}).

    ################################################################################################
    # The following are user-faced interface. The three abstract method of Collection's interface
    # are marked as non-abstract to allow instantiation in static checking, but still must be
    # implemented by a subclass. The rest of the interface have default implementations depending on
    # the three, but may also be override if a subclass implementation has a better algorithm, in
    # which case they must be explicitly overriden instead of provided by another base class, so
    # that we never accidentally override them to some wrong behaviour (only checked in runtime).

    ###########################
    # collections.abc.Container
    ###########################
    # NOTE: As a general interface, any object can be tested with `in`, not just int. This enables
    #       proper testing when a superclass of int is passed in and happens to be int.
    def __contains__(self, item: object) -> bool:
        """Test whether a variable is in the scope, for `in` and `not in` operators.

        Args:
            item (object): The item (any object) to test, but only int values are meaningful \
                variable ids.

        Returns:
            bool: Whether the variable is in this scope.
        """
        # With checks in register(), this raise should not be hit. Same for the following two.
        raise NotImplementedError("This should not be reached.")

    ##########################
    # collections.abc.Iterable
    ##########################
    def __iter__(self) -> Iterator[int]:
        """Iterate over the scope variables in the order of id, for conversion to other containers.

        Returns:
            Iterator[int]: The iterator over the scope (sorted).
        """
        raise NotImplementedError("This should not be reached.")

    #######################
    # collections.abc.Sized
    #######################
    def __len__(self) -> int:
        """Get the length (number of variables) of the scope, for len() as well as bool().

        Returns:
            int: The number of variables in the scope.
        """
        raise NotImplementedError("This should not be reached.")

    ############################
    # collections.abc.Collection
    ############################
    # Collection = Sized Iterable Container

    ##########################
    # collections.abc.Hashable
    ##########################
    def __hash__(self) -> int:
        """Get the hash value of the scope, for use as dict/set keys.

        The same scope (same set of variables) always has the same hash value.

        Returns:
            int: The hash value.
        """
        return hash(tuple(self))

    ################
    # Total Ordering
    ################
    # NOTE: As a general interface, any object can be tested with `==`, not just Scope. This enables
    #       proper testing when a superclass of Scope is passed in and happens to be Scope.
    def __eq__(self, other: object) -> bool:
        """Test equality between scopes, for == and != operators.

        Two scopes are equal when they contain the same set of variables, with hash also the same.

        Args:
            other (object): The other object to compare with, only Scope instances are meaningful.

        Returns:
            bool: Whether self == other.
        """
        return isinstance(other, Scope) and tuple(self) == tuple(other)

    # __ne__ automatically delegates to __eq__.

    # NOTE: Here we abuse the operators: < and > are for ordering, <= and >= are for subset test.
    def __lt__(self, other: "Scope") -> bool:
        """Compare scopes for ordering, for < operator.

        It is guaranteed that exactly one of a == b, a < b, a > b is True. Can be used for sorting \
        and the order is guaranteed to be always stable.

        Two scopes compare by the following:
            - If the lengths are different, the shorter one is smaller;
            - If of same length, the one with the smallest non-shared variable id is smaller;
            - They should be the same scope if the above cannot compare.

        Args:
            other (Scope): The other scope to compare with.

        Returns:
            bool: Whether self < other.
        """
        return len(self) < len(other) or len(self) == len(other) and tuple(self) < tuple(other)

    def __gt__(self, other: "Scope") -> bool:
        """Compare scopes for ordering, for > operator.

        a > b is defined as b < a, so that the reflection relationship holds.

        It is guaranteed that exactly one of a == b, a < b, a > b is True.

        Args:
            other (Scope): The other scope to compare with.

        Returns:
            bool: Whether self > other.
        """
        return NotImplemented  # Delegate to __lt__ by default.

    ################
    # Subset Testing
    ################
    # NOTE: Here we abuse the operators: < and > are for ordering, <= and >= are for subset test.
    def __le__(self, other: "Scope") -> bool:
        """Test whether self is a subset (or equal) of other.

        It is guaranteed that (a == b) <=> (a <= b and a >= b).

        Args:
            other (Scope): The other scope to test.

        Returns:
            bool: Whether self ⊆ other.
        """
        return frozenset(self) <= frozenset(other)

    def __ge__(self, other: "Scope") -> bool:
        """Test whether self is a superset (or equal) of other.

        a >= b is defined as b <= a, so that the reflection relationship holds.

        It is guaranteed that (a == b) <=> (a <= b and a >= b).

        Args:
            other (Scope): The other scope to test.

        Returns:
            bool: Whether self ⊇ other.
        """
        return NotImplemented  # Delegate to __lt__ by default.

    ########################
    # Union and Intersection
    ########################
    # NOTE: It's possible to accept any Iterable[int] as other, but we enforce Scope here.
    def __and__(self, other: "Scope") -> "Scope":
        """Get the intersection of two scopes, for & operator.

        Args:
            other (Scope): The other scope to take intersection with.

        Returns:
            Scope: The intersection.
        """
        return Scope(frozenset(self) & frozenset(other))

    def __or__(self, other: "Scope") -> "Scope":
        """Get the union of two scopes, for | operator.

        Args:
            other (Scope): The other scope to take union with.

        Returns:
            Scope: The union.
        """
        return Scope(frozenset(self) | frozenset(other))

    # DISABLE: This is a hack that self goes as the first of scopes, so that self.union(...) and
    #          Scope.union(...) both work, even when ... is empty.
    # pylint: disable-next=no-self-argument
    def union(*scopes: "Scope") -> "Scope":
        """Take the union over multiple scopes, for use as reduction with n-ary | operator.

        Can be used as either self.union(...) or Scope.union(...).

        Args:
            *scopes (Scope): The other scopes to take union with.

        Returns:
            Scope: The union.
        """
        return Scope(frozenset().union(*scopes))
