from abc import abstractmethod
from typing import (
    Callable,
    ClassVar,
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    Type,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)
from typing_extensions import Never, Self  # FUTURE: in typing from 3.11


class Scope(Collection[int], Hashable):
    """An immutable container (Hashable Collection) for a set of int to represent the scope of a \
    node in the region graph or a unit in the circuit.

    A scope should always be a subset of range(num_vars), but for efficiency this is not checked.
    """

    # TODO: ???
    # NOTE: The following also serves as the API for Scope. Even the methods defined in the base
    #       class can be reused, they should be overriden below to explicitly define the methods.
    # TODO: convert to bitset, and then all methods will be useful.

    # We should use __new__ instead of __init__ because it's immutable.
    # NOTE: Subclasses should implement:
    #       - Reusing the scope passed in if it's of the self class.
    #       - Making use of scope to initialize the internal container.
    @abstractmethod
    def __new__(cls, scope: Iterable[int]) -> Self:
        """Construct the scope object.

        Args:
            scope (Iterable[int]): The scope as an iterable of variable ids. If already an object \
                of the same class, the object will be directly returned.

        Returns:
            Self: The Scope object.
        """
        # Here we need this super() call to correctly chain the super() from subclasses. If a
        # subclass has another base class, this __new__ may also be skipped in favor of the other.
        return super().__new__(cls)

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
    # must be implemented, while the rest have default implementations depending on them. The
    # non-abstract methods may be override if a subclass implementation has a better algorithm, but
    # they must be explicitly overriden instead of provided by another base class.
    # TODO: runtime check? static check?

    ###########################
    # collections.abc.Container
    ###########################
    # NOTE: As a general interface, any object can be tested with `in`, not just int. This enables
    #       proper testing when a superclass of int is passed in and happens to be int.
    @abstractmethod
    def __contains__(self, item: object) -> bool:
        """Test whether a variable is in the scope, for `in` and `not in` operators.

        Args:
            item (object): The item (any object) to test, but only int values are meaningful \
                variable ids.

        Returns:
            bool: Whether the variable is in this scope.
        """

    ##########################
    # collections.abc.Iterable
    ##########################
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Iterate over the scope variables in the order of id, for conversion to other containers.

        Returns:
            Iterator[int]: The iterator over the scope (sorted).
        """

    #######################
    # collections.abc.Sized
    #######################
    @abstractmethod
    def __len__(self) -> int:
        """Get the length (number of variables) of the scope, for len() as well as bool().

        Returns:
            int: The number of variables in the scope.
        """

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
        if not isinstance(other, Scope):
            return False
        return tuple(self) == tuple(other)

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

    # Always delegate to __lt__, no need to override.
    @final
    def __gt__(self, other: "Scope") -> bool:
        """Compare scopes for ordering, for > operator.

        a > b is defined as b < a, so that the reflection relationship holds.

        It is guaranteed that exactly one of a == b, a < b, a > b is True.

        Args:
            other (Scope): The other scope to compare with.

        Returns:
            bool: Whether self > other.
        """
        return NotImplemented

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

    # Always delegate to __le__, no need to override.
    @final
    def __ge__(self, other: "Scope") -> bool:
        """Test whether self is a superset (or equal) of other.

        a >= b is defined as b <= a, so that the reflection relationship holds.

        It is guaranteed that (a == b) <=> (a <= b and a >= b).

        Args:
            other (Scope): The other scope to test.

        Returns:
            bool: Whether self ⊇ other.
        """
        return NotImplemented

    ########################
    # Union and Intersection
    ########################
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
    def union(*scopes: "Scope") -> "Scope":  # pylint: disable=no-self-argument
        """Take the union over multiple scopes, for use as reduction with n-ary | operator.

        Can be used as either self.union(...) or Scope.union(...).

        Args:
            *scopes (Scope): The other scopes to take union with.

        Returns:
            Scope: The union.
        """
        return Scope(frozenset().union(*scopes))
