from abc import ABC, abstractmethod


class LayerLabel(ABC):
    """The label associated with a layer class. It can be as simple as a string
    or a more complex structured object that describes a layer.
    """

    @abstractmethod
    def __repr__(self) -> str:
        """A string representation of the layer label.

        Returns:
            str: Label of the layer.
        """
