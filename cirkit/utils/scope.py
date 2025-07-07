from collections.abc import Collection, Hashable, Iterable, Iterator


class Scope(Collection[int], Hashable):
    """An immutable container for a set of int to represent the scope of a node in the region
    graph or a layer in the circuit. Currently, it is implemented on top of a Python frozenset of
    integers.
    """

    def __init__(self, scope: Iterable[int] | None = None):
        """Initializes a scope object.

        Args:
            scope: The scope as an iterable of variable non-negative integer IDs.
                It can be None as to construct an empty scope.

        Returns:
            Scope: A scope.
        """
        self._set: frozenset[int] = frozenset(scope) if scope is not None else frozenset()

    def __repr__(self) -> str:
        """Generate the repr string of the scope, for repr().

        Returns:
            str: The str representation of the scope.
        """
        return f"Scope({repr(set(self))})"  # Scope({0, 1, ...}).

    ################################################################################################
    # The following are the methods used by users.
    ################################################################################################

    def __contains__(self, var: object) -> bool:
        """Test whether a variable is in the scope, for `in` and `not in` operators.

        Args:
            var: The variable non-negative ID to test.

        Returns:
            bool: Whether the variable is in this scope.
        """
        if not isinstance(var, int):
            return NotImplemented
        return var in self._set

    def __iter__(self) -> Iterator[int]:
        """Iterate over the scope variables in the order of id, for conversion to other containers.

        Returns:
            Iterator[int]: The iterator over the scope, that is sorted.
        """
        return iter(self._set)

    def __len__(self) -> int:
        """Get the length, i.e., the number of variables) of the scope.

        Returns:
            int: The number of variables in the scope.
        """
        return len(self._set)

    def __hash__(self) -> int:
        """Get the hash value of the scope, for use as dict/set keys.

        The same scope (same set of variables) always has the same hash value.

        Returns:
            int: The hash value.
        """
        return hash(self._set)

    def __eq__(self, other: object) -> bool:
        """Test equality between scopes, for == and != operators.

        Two scopes are equal when they contain the same set of variables, with hash also the same.

        Args:
            other: The other scope to compare with.

        Returns:
            bool: Whether self == other.
        """
        if not isinstance(other, Scope):
            return NotImplemented
        return self._set == other._set

    def __lt__(self, other: "Scope") -> bool:
        """Test whether self is a subset (strictly) of other.

        Args:
            other: The other scope to test.

        Returns:
            bool: Whether self ⊆ other.
        """
        return self._set < other._set

    def __gt__(self, other: "Scope") -> bool:
        """Test whether self is a superset (strictly) of other.

        Args:
            other: The other scope to test.

        Returns:
            bool: Whether self ⊇ other.
        """
        return self._set > other._set

    def __le__(self, other: "Scope") -> bool:
        """Test whether self is a subset (or equal) of other.

        It is guaranteed that (a == b) <=> (a <= b and a >= b).

        Args:
            other: The other scope to test.

        Returns:
            bool: Whether self ⊆ other.
        """
        return self._set <= other._set

    def __ge__(self, other: "Scope") -> bool:
        """Test whether self is a superset (or equal) of other.

        Args:
            other: The other scope to test.

        Returns:
            bool: Whether self ⊇ other.
        """
        return self._set >= other._set

    def __and__(self, other: "Scope") -> "Scope":
        """Get the intersection of two scopes, for & operator.

        Args:
            other: The other scope to take intersection with.

        Returns:
            Scope: The intersection.
        """
        return Scope(self._set & other._set)

    def __or__(self, other: "Scope") -> "Scope":
        """Get the union of two scopes, for | operator.

        Args:
            other: The other scope to take union with.

        Returns:
            Scope: The union.
        """
        return Scope(self._set | other._set)

    # pylint: disable-next=no-self-argument
    def union(*scopes: "Scope") -> "Scope":
        """Take the union over multiple scopes, for use as reduction with n-ary | operator.

        Can be used as either self.union(...) or Scope.union(...).

        Args:
            *scopes: The other scope to take union with.

        Returns:
            Scope: The union.
        """
        sets = tuple(s._set for s in scopes)  # pylint: disable=protected-access
        # Populate the scope by taking the unions of the input scopes
        scope = Scope(frozenset().union(*sets))
        return scope

    def difference(self, other: "Scope") -> "Scope":
        """Take the difference w.r.t. another scope, i.e., the scope
        containing the variables that are not in the other scope.

        Args:
            other: The other scope to take the difference with.

        Returns:
            Scope: The difference between scopes.
        """
        return Scope(self._set.difference(other._set))  # pylint: disable=protected-access

    def __sub__(self, other: "Scope") -> "Scope":
        """Take the difference w.r.t. another scope, i.e., the scope
        containing the variables that are not in the other scope.

        Args:
            other: The other scope to take the difference with.

        Returns:
            Scope: The difference between scopes.
        """
        return self.difference(other)
