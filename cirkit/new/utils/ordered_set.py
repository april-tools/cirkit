from typing import Any, Collection, Dict, Iterable, Iterator, Protocol, TypeVar, final


# TODO: pylint issue? protocol are expected to have few public methods
class _SupportsDunderLT(Protocol):  # pylint: disable=too-few-public-methods
    # Disable: This is the only way to get a TypeVar for Protocol with __lt__. Another option, using
    #          Protocol[T_contra], will introduce much more ignores.
    def __lt__(self, other: Any, /) -> bool:  # type: ignore[misc]
        ...


T = TypeVar("T")
ComparableT = TypeVar("ComparableT", bound=_SupportsDunderLT)
# These, to be used as (mutable) Collection[T], can't be either covariant or contravariant:
# - Function arguments cannot be covariant, therefore nor mutable generic types;
#   - See explaination in https://github.com/python/mypy/issues/7049;
# - Containers can never be contravariant by nature.


# There's no need to inherit this.
@final
class OrderedSet(Collection[T]):
    """A mutable container of a set that preserves element ordering when iterated.

    The elements are required to support __lt__ comparison to make sort() work.

    This is designed for node (edge) lists in the graph data structure (incl. RG, SymbC, ...), but
    does not comply with all standard builtin container (list, set, dict) interface.

    The implementation relies on the order preservation of dict introduced in Python 3.7.
    """

    # NOTE: We can also inherit Reversible[T] and implement __reversed__ based on reverse(dict), but
    #       currently this feature is not needed.

    def __init__(self, *iterables: Iterable[T]) -> None:
        """Init class.

        Args:
            *iterables (Iterable[T]): The initial content, if provided any. Will be inserted in \
                the order given.
        """
        super().__init__()
        # The dict values are unused and always set to True.
        self._container: Dict[T, T] = {
            element: element for iterable in iterables for element in iterable
        }

    # Ignore: We should only test the element type.
    def __contains__(self, element: T) -> bool:  # type: ignore[override]
        """Test whether an element is contained in the set.

        Args:
            element (T): The element to test.

        Returns:
            bool: Whether the element exists.
        """
        return element in self._container

    def __iter__(self) -> Iterator[T]:
        """Iterate over the set in order.

        Returns:
            Iterator[T]: The iterator over the set.
        """
        return iter(self._container)

    def __len__(self) -> int:
        """Get the length (number of elements) of the set.

        Returns:
            int: The number of elements in the set.
        """
        return len(self._container)

    def append(self, element: T) -> bool:
        """Add an element to the end of the set, if it does not exist yet; otherwise no-op.

        New elements are always added at the end, and the order is preserved. Existing elements \
        will not be moved if appended again.

        Args:
            element (T): The element to append.

        Returns:
            bool: Whether the insertion actually happened at the end.
        """
        if element in self:
            return False  # Meaning it's a no-op.

        self._container[element] = element
        return True  # Meaning a successful append.

    def extend(self, elements: Iterable[T]) -> None:
        """Add elements to the end of the set, for each one in order that does not exist yet; \
        duplicates are skipped.

        The input iterable should have a deterministic order and the order is preserved.

        Args:
            elements (Iterable[T]): The elements to extend.
        """
        for element in elements:
            self.append(element)

    def sort(self: "OrderedSet[ComparableT]") -> "OrderedSet[ComparableT]":
        """Sort the set inplace and return self.

        The elements must support __lt__ comparison.

        It stably sorts the elements from the insertion order to the comparison order:
        - If a < b, a always precedes b in the sorted order;
        - If neither a < b nor b < a, the existing order is preserved.

        Returns:
            Self: The self object.
        """
        # This relies on that sorted() is stable.
        self._container = {element: element for element in sorted(self._container)}
        return self
