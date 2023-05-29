from abc import ABC, abstractmethod
from typing import Any

import networkx as nx


class RegionGraph(ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        super().__init__()
        self._graph = self._construct_graph(*args, **kwargs)  # type: ignore[misc]

    @staticmethod
    @abstractmethod
    def _construct_graph(*args: Any, **kwargs: Any) -> nx.DiGraph:  # type: ignore[misc]
        pass
