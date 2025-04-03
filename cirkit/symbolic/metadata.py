from collections.abc import MutableMapping
from typing import Any, Iterator, cast


class LayerMetadata(MutableMapping):
    """The metadata for a symbolic layer. It acts as a recursive
    defaultdict object where an empty dict is always used when a
    key is not available. Keys are always strings.
    """

    def __init__(self):
        """Initialize the layer metadata as an empty dictionary."""
        self._map: dict = {}

    def items(self) -> Iterator[tuple[str, Any]]:
        """
        Returns the items of the layer metadata.

        Returns:
            Iterator[tuple[str, Any]]: An iterable where each element
                is a tuple with the key and its value.
        """
        return cast(Iterator[tuple[str, Any]], self._map.items())

    def __copy__(self) -> "LayerMetadata":
        """Shallow copy the metadata.

        Returns:
            LayerMetadata: A shallow copy of the metadata.
        """
        metadata = LayerMetadata()
        metadata._map = self._map.copy()
        return metadata

    def __iter__(self) -> Iterator[str]:
        """
        Returns the iterable over the keys in the metadata.

        Returns:
            Iterable[str]: The iterable of keys.
        """
        return iter(self._map)

    def __setitem__(self, k: str, v: Any):
        """
        Sets an item in the metadata.

        Args:
            k: The key of the item.
            v: The value of the item.
        """
        self._map[k] = v

    def __delitem__(self, k: str):
        """
        Deletes an item from the metadata.

        Args:
            k: The key of the item to delete.
        """
        del self._map[k]

    def __len__(self) -> int:
        """
        Returns the number of elements in the metadata.

        Returns:
            int: Elements in the metadata.
        """
        return len(self._map)

    def __getitem__(self, k: str) -> Any:
        """
        Returns the element at key k if it has been stored in the metadata.
        If it has not been stored yet, an empty LayerMetadata object is
        stored at that key and returned instead.

        Args:
            k: The key of the item.

        Returns:
            Any: The element stored as the key k.
        """
        if k not in self._map:
            # automatically instantiate sub metadata object for k
            self[k] = {}
        return self._map[k]

    def __contains__(self, k: str) -> bool:
        """
        Checks whether the layer metadata contains a key.
        If a key has not been set or retrieved yet, it is considered
        as not existing.

        Args:
            k: The key of the metadata.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return k in self._map
