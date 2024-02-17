from typing import Collection, Iterable, Iterator, Protocol, TypeVar, final
from typing_extensions import Self  # FUTURE: in typing from 3.11


# We use this Protocol to construct a TypeVar for classes with __lt__. Ref: typeshed.
# TODO: pylint issue? protocol are expected to have few public methods
# pylint: disable-next=too-few-public-methods
class _SupportsDunderLT(Protocol):
    def __lt__(self, other: Self) -> bool:  # At least support comparison with self type.
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

    def __init__(self, *iterables: Iterable[T]) -> None:
        """Init class.

        Args:
            *iterables (Iterable[T]): The initial content, if provided any. Will be inserted in \
                the order given.
        """
        super().__init__()
        self._container = {element: element for iterable in iterables for element in iterable}

    ###########################
    # collections.abc.Container
    ###########################
    # NOTE: As a general interface, any object can be tested with `in`, not just T. This enables
    #       proper testing when a superclass of T is passed in and happens to be T.
    def __contains__(self, item: object) -> bool:
        """Test whether an item (any object) is contained in the set.

        Args:
            item (object): The item to test. Any object can be tests, but only those of type T \
                make sense.

        Returns:
            bool: Whether the item exists.
        """
        return item in self._container

    ##########################
    # collections.abc.Iterable
    ##########################
    def __iter__(self) -> Iterator[T]:
        """Iterate over the set in order.

        Returns:
            Iterator[T]: The iterator over the set.
        """
        return iter(self._container)

    #######################
    # collections.abc.Sized
    #######################
    def __len__(self) -> int:
        """Get the length (number of elements) of the set.

        Returns:
            int: The number of elements in the set.
        """
        return len(self._container)

    ############################
    # collections.abc.Collection
    ############################
    # Collection = Sized Iterable Container

    ########################
    # Part of list interface
    ########################
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
