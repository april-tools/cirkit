from typing import FrozenSet, Iterable, Iterator, Union, final
from typing_extensions import Self  # FUTURE: in typing from 3.11


# We mark this final so that Scope==Self in typing. Also there's no need to inherit this.
@final
class Scope(FrozenSet[int]):
    """An immutable container (Hashable Collection) of int to represent the scope of a unit in a \
    circuit.

    Scopes should always be subsets of range(num_vars), but for efficiency this is not checked.
    """

    # NOTE: The following also serves as the API for Scope. Even the methods defined in the base
    #       class can be reused, they should be overriden below to explicitly define the methods.
    # TODO: convert to bitset, and then all methods will be useful.

    # We should use __new__ instead of __init__ because it's immutable.
    def __new__(cls, scope: Union["Scope", Iterable[int]]) -> Self:
        """Create the scope.

        Args:
            scope (Union[Scope, Iterable[int]]): The scope as an interable of variable ids. If \
                already a Scope object, the object passed in will be directly returned.

        Returns:
            Self: The Scope object.
        """
        if isinstance(scope, Scope):  # Saves a copy.
            return scope
        # TODO: mypy bug? asking for Iterable[_T_co] but it's already FrozenSet[int]
        return super().__new__(cls, scope)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Generate the repr string of the scope, for repr().

        Returns:
            str: The str representation of the scope.
        """
        return f"Scope({repr(set(self))})"  # Scope({0, 1, ...}).

    ###########################
    # collections.abc.Container
    ###########################
    # IGNORE: We should only test int.
    def __contains__(self, var: int) -> bool:  # type: ignore[override]
        """Test whether a variable is in the scope, for `in` and `not in` operators.

        Args:
            var (int): The variable id to test.

        Returns:
            bool: Whether the variable is in this scope.
        """
        return super().__contains__(var)

    ##########################
    # collections.abc.Iterable
    ##########################
    def __iter__(self) -> Iterator[int]:
        """Iterate over the scope variables in the order of id, for conversion to other containers.

        Returns:
            Iterator[int]: The iterator over the scope (sorted).
        """
        return iter(sorted(super().__iter__()))  # FrozenSet is not sorted.

    #######################
    # collections.abc.Sized
    #######################
    # DISABLE: We require explicit definition.
    # pylint: disable-next=useless-parent-delegation
    def __len__(self) -> int:
        """Get the length (number of variables) of the scope, for len() as well as bool().

        Returns:
            int: The number of variables in the scope.
        """
        return super().__len__()

    ############################
    # collections.abc.Collection
    ############################
    # Collection = Sized Iterable Container

    ##########################
    # collections.abc.Hashable
    ##########################
    def __hash__(self) -> int:
        """Get the hash value of the scope, for use as dict/set keys.

        The same scope always has the same hash value.

        Returns:
            int: The hash value.
        """
        return super().__hash__()

    ################
    # Total Ordering
    ################
    # IGNORE: We should only compare scopes.
    def __eq__(self, other: "Scope") -> bool:  # type: ignore[override]
        """Test equality between scopes, for == and != operators.

        Two scopes are equal when they contain the same set of variables.

        Args:
            other (Scope): The other scope to compare with.

        Returns:
            bool: Whether self == other.
        """
        return super().__eq__(other)

    # IGNORE: We should only compare scopes.
    def __lt__(self, other: "Scope") -> bool:  # type: ignore[override]
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

    # IGNORE: We should only compare scopes.
    def __gt__(self, other: "Scope") -> bool:  # type: ignore[override]
        """Compare scopes for ordering, for > operator.

        a > b is defined as b < a, so that the reflection relationship holds.

        It is guaranteed that exactly one of a == b, a < b, a > b is True.

        Args:
            other (Scope): The other scope to compare with.

        Returns:
            bool: Whether self > other.
        """
        return other < self

    #############
    # Subset Test
    #############
    # NOTE: Here we abuse the operators: < and > are for ordering, <= and >= are for subset test.
    # IGNORE: We should only compare scopes.
    def __le__(self, other: "Scope") -> bool:  # type: ignore[override]
        """Test whether self is a subset (or equal) of other.

        Args:
            other (Scope): The other scope to test.

        Returns:
            bool: Whether self ⊆ other.
        """
        return super().__le__(other)

    # IGNORE: We should only compare scopes.
    def __ge__(self, other: "Scope") -> bool:  # type: ignore[override]
        """Test whether self is a superset (or equal) of other.

        Args:
            other (Scope): The other scope to test.

        Returns:
            bool: Whether self ⊇ other.
        """
        return other <= self

    ########################
    # Union and Intersection
    ########################
    # IGNORE: We should only intersect scopes.
    def __and__(self, other: "Scope") -> "Scope":  # type: ignore[override]
        """Get the intersection of two scopes, for & operator.

        Args:
            other (Scope): The other scope to take intersection with.

        Returns:
            Scope: The intersection.
        """
        return Scope(super().__and__(other))

    # IGNORE: We should only union scopes.
    def __or__(self, other: "Scope") -> "Scope":  # type: ignore[override]
        """Get the union of two scopes, for | operator.

        Args:
            other (Scope): The other scope to take union with.

        Returns:
            Scope: The union.
        """
        return Scope(super().__or__(other))

    # DISABLE: This is a hack that self goes as the first of scopes, so that self.union(...) and
    #          Scope.union(...) both work, even when ... is empty.
    # IGNORE: We should only union scopes.
    # pylint: disable-next=no-self-argument
    def union(*scopes: "Scope") -> "Scope":  # type: ignore[override]
        """Take the union over multiple scopes, for use as n-ary | operator.

        Can be used as either self.union(...) or Scope.union(...).

        Args:
            *scopes (Scope): The other scopes to take union with.

        Returns:
            Scope: The union.
        """
        return Scope(frozenset().union(*scopes))
